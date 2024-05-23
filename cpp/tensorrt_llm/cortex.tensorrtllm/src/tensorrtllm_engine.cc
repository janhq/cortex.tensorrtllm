#include "tensorrtllm_engine.h"

#include <filesystem>
#include <thread>
#include <utility>

#include "trantor/utils/Logger.h"
#include "utils/tensorrtllm_utils.h"
#include "nlohmann/json.hpp"


constexpr const int k200OK = 200;
constexpr const int k400BadRequest = 400;
constexpr const int k409Conflict = 409;
constexpr const int k500InternalServerError = 500;

TensorrtllmEngine::~TensorrtllmEngine() {}

void TensorrtllmEngine::LoadModel(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
    
  LoadModelImpl(tensorrtllm::model::fromJson(json_body), std::move(callback));
}

void TensorrtllmEngine::HandleChatCompletion(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  HandleInferenceImpl(tensorrtllm::inferences::fromJson(json_body), std::move(callback));
}

void TensorrtllmEngine::Destroy(
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  DestroyImpl(std::move(callback));
}  

// #######################
// ### IMPLEMENTATION ####
// #######################

bool TensorrtllmEngine::LoadModelImpl(
    tensorrtllm::model::LoadModelRequest&& request,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  std::filesystem::path engine_dir, config_json_file, tokenizer_model_file;
  if (!tensorrtllm_utils::ValidateLoadModelRequest(request, callback) ||
      !tensorrtllm_utils::ValidateLoadModelFiles(request.engine_path, engine_dir,
                                                 config_json_file, tokenizer_model_file)) {    
    return false;
  }

  this->ai_prompt     = request.ai_prompt;
  this->user_prompt   = request.user_prompt;
  this->system_prompt = request.system_prompt;
  server_map_[request.model_id].q 
      = std::make_unique<trantor::ConcurrentTaskQueue>(request.n_parallel, request.model_id);

  logger = std::make_shared<TllmLogger>();
  logger->setLevel(nvinfer1::ILogger::Severity::kINFO);
  // Fixed settings
  std::string const modelName = "mistral";
  initTrtLlmPlugins(logger.get());
  // Load model configuration

  cortex_tokenizer = std::make_unique<Tokenizer>(tokenizer_model_file.string());
  LOG_INFO << "Loaded tokenizer";

  auto const gpt_json_conf = GptJsonConfig::parse(config_json_file);
  auto config = gpt_json_conf.getModelConfig();
  model_config = std::make_unique<ModelConfig>(config);
  auto const world_config = WorldConfig::mpi(1, gpt_json_conf.getTensorParallelism(), 
                                                                gpt_json_conf.getPipelineParallelism());
  auto const trtllm_engine_path = engine_dir / gpt_json_conf.engineFilename(world_config, modelName);
  LOG_INFO << "TRT-LLM Engine Path : " << trtllm_engine_path.string();

  // Currently doing fixed session config
  session_config.maxBatchSize = batch_size;
  session_config.maxBeamWidth = 1; // Fixed for simplicity
  session_config.maxSequenceLength = request.ctx_len;
  session_config.cudaGraphMode = true; // Fixed for simplicity

  // Init gptSession
  gpt_session = std::make_unique<GptSession>(session_config, *model_config,
                                             world_config, engine_dir.string(), logger);

  // Model loaded successfully
  Json::Value json_resp;
  Json::Value status_resp;
  json_resp["message"] = "Model loaded successfully";
  status_resp["status_code"] = k200OK;
  callback(std::move(status_resp), std::move(json_resp));
  return true;
}

GenerationInput::TensorPtr TensorrtllmEngine::GetTensorChatMLStopWordList() {
  // Extend with -1 for increased length
  std::vector<int32_t> stop_words_tokens = {321, 28730, 416, 2, 32000, 3, 4, 5, -1, -1}; 
  return gpt_session->getBufferManager().copyFrom(
      stop_words_tokens, 
      ITensor::makeShape({1, 2, 5}), 
      MemoryType::kGPU);
}

GenerationInput TensorrtllmEngine::CreateGenerationInput(std::vector<int32_t> input_ids_host) {
  int input_len = input_ids_host.size();
  std::vector<int32_t> input_lengths_host(batch_size, input_len);
  GenerationInput::TensorPtr input_tensor_len = gpt_session->getBufferManager().copyFrom(
      input_lengths_host,
      ITensor::makeShape({batch_size}), 
      MemoryType::kGPU);
  GenerationInput::TensorPtr input_tensor_ids = gpt_session->getBufferManager().copyFrom(
      input_ids_host, 
      ITensor::makeShape({batch_size, input_len}), 
      MemoryType::kGPU);
  GenerationInput generationInput{0, 0, input_tensor_ids, input_tensor_len, model_config->usePackedInput()};

  generationInput.stopWordsList = GetTensorChatMLStopWordList();
  return generationInput;
}

GenerationOutput TensorrtllmEngine::CreateGenerationOutput() {
  GenerationOutput generationOutput{
      gpt_session->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
      gpt_session->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};
  return generationOutput;
}

void InferenceThread(std::shared_ptr<InferenceState> infer_state, std::vector<int32_t> input_ids_host,
                     const std::function<void(Json::Value&&, Json::Value&&)>& callback, TensorrtllmEngine* self, 
                     SamplingConfig sampling_config, int input_len, int output_len) {

  // Input preparation
  GenerationInput generation_input   = self->CreateGenerationInput(input_ids_host);
  GenerationOutput generation_output = self->CreateGenerationOutput();

  // Define the callback to stream each generated token
  generation_output.onTokenGenerated = [&infer_state, input_len, output_len, self, &generation_output](
      GenerationOutput::TensorPtr const& output_ids, 
      SizeType step, 
      bool finished) {
    // Assuming the shape of outputIds tensor is (1, 1, 160), where 160 is the number of tokens
    int output_tensor_len = output_ids->getShape().d[2]; // Get the length of output IDs based on the tensor shape
    // Copy output IDs from GPU to host for printing
    std::vector<int32_t> output_ids_host(output_tensor_len);
    self->gpt_session->getBufferManager().copy(*output_ids, output_ids_host.data(), MemoryType::kCPU);
    // Find the last non-zero value in the output IDs starting from the end of the input sequence
    std::vector<int> output_ids_host_decode(output_ids_host.begin() + input_len, output_ids_host.end());

    auto remove_id = [](std::vector<int>& vec, int id) {
      vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
    };
    remove_id(output_ids_host_decode, 0);
    remove_id(output_ids_host_decode, 32000);
    remove_id(output_ids_host_decode, 32001);
    std::string text = self->cortex_tokenizer->Decode(output_ids_host_decode);

    if (infer_state->prev_pos >= 0 && infer_state->prev_pos < text.size()) {
      // Valid prevPos, proceed with slicing the string from prevPos to the end
      std::string stringTok(text.begin() + infer_state->prev_pos, text.end());
      std::lock_guard<std::mutex> guard(infer_state->queue_mutex); // Protect access with a lock
      infer_state->texts_to_stream.push(stringTok);
    } else if (infer_state->prev_pos >= text.size()) {
      infer_state->prev_pos = text.size();
    }
    infer_state->prev_pos = text.size();
    if (finished) {
      std::lock_guard<std::mutex> guard(infer_state->queue_mutex); // Protect access with a lock
      infer_state->texts_to_stream.push("[DONE]");
      return;
    }
    return;
  };
  // The rest of the logic inside the `chat_completion` remains unchanged...
  // After finishing the setup, call the inference logic
  self->gpt_session->generate(generation_output, generation_input, sampling_config);
}

void TensorrtllmEngine::GetFormattedInputFromMessages(const Json::Value& messages, std::string& formatted_input) {
  // Format the input from user
  for (auto const& message : messages) {
    std::string input_role = message["role"].asString();
    std::string role;
    if (input_role == "user") {
      role = user_prompt;
      std::string content = message["content"].asString();
      formatted_input += role + content;
    }
    else if (input_role == "assistant") {
      role = ai_prompt;
      std::string content = message["content"].asString();
      formatted_input += role + content;
    }
    else if (input_role == "system") {
      role = system_prompt;
      std::string content = message["content"].asString();
      formatted_input = role + content + formatted_input;
    }
    else {
      role = input_role;
      std::string content = message["content"].asString();
      formatted_input += role + content;
    }
  }
  formatted_input += ai_prompt;
}

bool HandleMatch(std::string const& rawText, std::shared_ptr<InferenceState> inferState) {
  if (inferState->isComplete()) {
    return false;
  }
  if (inferState->stop_word_match_len == 0) {
    if (rawText.find('<') != std::string::npos) { // Found "<" anywhere in the text
      inferState->stop_word_match_len++; // Move to next state
      inferState->prev_text = rawText;
      return true;
    }
  }
  else if (rawText == inferState->sequence[inferState->stop_word_match_len]) {
    inferState->stop_word_match_len++; // Move to next state
    inferState->prev_text = rawText;
    return true;
  }
  else if (inferState->stop_word_match_len > 0 && rawText == inferState->sequence[0]) {
    inferState->stop_word_match_len = 1; // Restart from first match if sequence breaks but matches start
    inferState->prev_text = rawText;
    return true;
  }
  else {
    inferState->reset();
    return false; // Reset to start if sequence breaks
  }
  return false;
}

void TensorrtllmEngine::HandleInferenceImpl(
    tensorrtllm::inferences::ChatCompletionRequest&& request,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  if (!tensorrtllm_utils::ValidateHandleChatCompletionRequest(request, callback)) {
    return;
  }

  if (server_map_.find(request.model_id) != server_map_.end()) {
    LOG_ERROR << "Model " << request.model_id << " was not loaded!";
    return;
  }
  ServerInfo& server_info = server_map_[request.model_id];

  std::string formatted_input = pre_prompt;
  nlohmann::json data;
  // data["stream"] = request.stream;
  // data["n_predict"] = request.max_tokens;
  data["presence_penalty"] = request.presence_penalty;

  Json::Value messages = request.messages;
  GetFormattedInputFromMessages(messages, formatted_input);  

  std::shared_ptr<InferenceState> infer_state = std::make_shared<InferenceState>();
  std::vector<int32_t> input_ids_host = cortex_tokenizer->Encode(formatted_input);
  int input_len = input_ids_host.size();
  int outputLen = request.max_tokens - input_len;

  // Create sampling config
  SamplingConfig sampling_config{1};
  sampling_config.temperature       = std::vector{request.temperature};
  sampling_config.randomSeed        = std::vector{static_cast<uint64_t>(42ull)};
  sampling_config.topK              = std::vector{40};
  sampling_config.topP              = std::vector{request.top_p};
  sampling_config.minLength         = std::vector{outputLen};
  sampling_config.repetitionPenalty = std::vector{request.frequency_penalty};

  // Input preparation
  std::thread inference_thread(InferenceThread, infer_state, input_ids_host, 
                               callback, this, sampling_config, input_len, outputLen);
  inference_thread.detach(); // Detach the thread to allow it to run independently

  server_info.q->runTaskInQueue([cb = std::move(callback), this, infer_state]() {
    while(true) { // Continuously check if the queue is not empty
      std::unique_lock<std::mutex> lock(infer_state->queue_mutex); // Lock the queue for exclusive access
      if (!infer_state->texts_to_stream.empty()) {
        std::string raw_text = infer_state->texts_to_stream.front();
        infer_state->texts_to_stream.pop();
        if (HandleMatch(raw_text, infer_state) && raw_text != "[DONE]") {
          continue;
        }

        Json::Value resp_data;
        Json::Value status;
        std::string text_to_stream;
        if (raw_text == "[DONE]") {
          text_to_stream
              = "data: " + tensorrtllm_utils::CreateReturnJson(tensorrtllm_utils::GenerateRandomString(20),
                                                               "_", "", "stop")
              + "\n\n" + "data: [DONE]" + "\n\n";
          infer_state->is_finished = true;
        } else {
          text_to_stream
            = "data: " + tensorrtllm_utils::CreateReturnJson(tensorrtllm_utils::GenerateRandomString(20),
                                                             "_", raw_text) + "\n\n";
        }
        lock.unlock(); // Unlock as soon as possible
        infer_state->prev_text = raw_text;
        resp_data["data"] = text_to_stream;
        status["is_done"] = false;
        status["has_error"] = false;
        status["is_stream"] = true;
        status["status_code"] = k200OK;
        cb(std::move(status), std::move(resp_data));
      } else {
        // If the queue is empty, release the lock and wait before trying again
        lock.unlock();
      }
    }
  });
}

void TensorrtllmEngine::DestroyImpl(std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  LOG_INFO << "Program is exitting, goodbye!";
  exit(0);
  return;
}

extern "C" {
EngineI* get_engine() {
  return new TensorrtllmEngine();
}
}
