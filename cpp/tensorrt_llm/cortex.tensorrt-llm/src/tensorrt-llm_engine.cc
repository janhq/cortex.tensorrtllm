#include "tensorrt-llm_engine.h"
#include "models/chat_completion_request.h"
#include "nlohmann/json.hpp"

#include <trantor/utils/Logger.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "cpp-tiktoken/encoding.h"  //include to use tiktoken
#include "json/writer.h"
#include "src/models/load_model_request.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "utils/tensorrt-llm_utils.h"

using json = nlohmann::json;
using namespace tensorrtllm;

namespace {
constexpr const int k200OK = 200;
constexpr const int k400BadRequest = 400;
constexpr const int k409Conflict = 409;
constexpr const int k500InternalServerError = 500;
constexpr const int kFileLoggerOption = 0;

// '<', '|', 'im', '_', 'end', '|', '>', '</s>', '<|im_end|>'
const std::list<std::vector<int32_t>> kOpenhermesStopWords = {
    {28789, 28766, 321, 28730, 416, 28766, 28767},
    {2},
    {32000}};
const std::string kOhUserPrompt = "<|im_end|>\n<|im_start|>user\n";
const std::string kOhAiPrompt = "<|im_end|>\n<|im_start|>assistant\n";
const std::string kOhSystemPrompt = "<|im_start|>system\n";

// '</s>'
const std::list<std::vector<int32_t>> kMistral_V0_3_StopWords = {{2}};

enum class OpenhermesTemplate : int32_t { kImEnd = 32000, kImStart = 32001 };

enum class MistralTemplate : int32_t {
  kBos = 1,
  kEos = 2,
  kBeginInst = 3,
  kEndInst = 4
};

enum class Llama3Template : int32_t {
  kBeginOfText = 128000,
  kEndOfText = 128001,
  kEndOfTurn = 128009,
  kStartHeaderId = 128006,
  kEndHeaderId = 128007,
  kParagraph = 271
};

// "<|end_of_text|>", "<|eot_id|>"
const std::list<std::vector<int32_t>> Llama3StopWords = {{128001}, {128009}};

// TODO(sang) This is fragile, just a temporary solution. Maybe can use a config file or model architect, etc...
bool IsOpenhermes(const std::string& s) {
  if (s.find("mistral") != std::string::npos ||
      s.find("Mistral") != std::string::npos) {
    return false;
  }
  return true;
}
ModelType GetModelType(const std::string& s) {
  if (s.find("Llama3") != std::string::npos ||
      s.find("llama3") != std::string::npos) {
    return ModelType::kLlama3;
  } else if (s.find("mistral") != std::string::npos ||
             s.find("Mistral") != std::string::npos) {
    return ModelType::kMistral;
  } else {
    return ModelType::kOpenHermes;
  }
}

std::list<std::vector<int32_t>> GetStopWords(ModelType model_type) {
  switch (model_type) {
    case ModelType::kLlama3:
      return Llama3StopWords;
    case ModelType::kMistral:
      return kMistral_V0_3_StopWords;
    default:
      return kOpenhermesStopWords;
  }
}

void RemoveSpecialTokens(std::vector<int32_t>& v, ModelType model_type) {
  auto remove_id = [](std::vector<int>& vec, int id) {
    vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
  };
  switch (model_type) {
    case ModelType::kLlama3:
      remove_id(v, static_cast<int32_t>(Llama3Template::kEndOfText));
      remove_id(v, static_cast<int32_t>(Llama3Template::kEndOfTurn));
      break;
    case ModelType::kMistral:
      remove_id(v, static_cast<int32_t>(MistralTemplate::kEos));
      break;
    default:
      remove_id(v, static_cast<int32_t>(OpenhermesTemplate::kImEnd));
      remove_id(v, static_cast<int32_t>(OpenhermesTemplate::kImStart));
      break;
  }
}
}  // namespace
TensorrtllmEngine::TensorrtllmEngine(int log_option) {
  trantor::Logger::setLogLevel(trantor::Logger::kError);
  if (log_option == kFileLoggerOption) {
    std::filesystem::create_directories(log_folder);
    asynce_file_logger_ = std::make_unique<trantor::AsyncFileLogger>();
    asynce_file_logger_->setFileName(log_base_name);
    asynce_file_logger_->startLogging();
    trantor::Logger::setOutputFunction(
        [&](const char* msg, const uint64_t len) {
          asynce_file_logger_->output(msg, len);
        },
        [&]() { asynce_file_logger_->flush(); });
    asynce_file_logger_->setFileSizeLimit(max_log_file_size);
  }
}

TensorrtllmEngine::~TensorrtllmEngine() {
  model_loaded_ = false;
  if (res_thread_ && res_thread_->joinable()) {
    res_thread_->join();
  }
  asynce_file_logger_.reset();
}

void RemoveId(std::vector<int>& vec, int id) {
  vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
}

bool HandleMatch(std::string const& rew_text, InferenceState* infer_state,
                 std::function<void(Json::Value&&, Json::Value&&)> cb,
                 ModelType model_type) {
  if (infer_state->IsComplete(model_type)) {
    return false;
  }
  if (infer_state->GetStopMatchLen() == 0) {
    if ((model_type == ModelType::kOpenHermes &&
         rew_text.find('<') != std::string::npos) ||
        (model_type != ModelType::kOpenHermes &&
         rew_text.find('[') != std::string::npos)) {
      infer_state->AddStopMatch();  // Move to next state
      return true;
    }
  } else if (rew_text == infer_state->GetSequence(
                             model_type, infer_state->GetStopMatchLen())) {
    infer_state->AddStopMatch();  // Move to next state
    return true;
  } else if (infer_state->GetStopMatchLen() > 0 &&
             rew_text == infer_state->GetSequence(model_type, 0u)) {
    infer_state
        ->ResetStopMatch();  // Restart from first match if sequence breaks but matches start
    return true;
  } else {
    infer_state->Reset();
    return false;  // Reset to start if sequence breaks
  }
  return false;
}

inline std::string GetModelId(const Json::Value& json_body) {
  // First check if model exists in request
  if (!json_body["model"].isNull()) {
    return json_body["model"].asString();
  } else if (!json_body["model_alias"].isNull()) {
    return json_body["model_alias"].asString();
  }

  // We check model_path for loadmodel request
  auto input = json_body["model_path"];
  if (!input.isNull()) {
    auto s = input.asString();
    std::replace(s.begin(), s.end(), '\\', '/');
    auto const pos = s.find_last_of('/');
    return s.substr(pos + 1);
  }
  return {};
}

bool TensorrtllmEngine::CheckModelLoaded(
    std::function<void(Json::Value&&, Json::Value&&)>& callback) {
  if (!model_loaded_) {
    LOG_WARN << "Model is not loaded yet";
    Json::Value json_resp;
    json_resp["message"] =
        "Model has not been loaded, please load model into cortex.tensorrt-llm";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k409Conflict;
    callback(std::move(status), std::move(json_resp));
    return false;
  }
  return true;
}

std::vector<int> TensorrtllmEngine::EncodeHeaderLlama3(
    const std::string& role) {
  std::vector<int> tokens = {};
  tokens.push_back(static_cast<int32_t>(
      Llama3Template::kStartHeaderId));  // <|start_header_id|>
  auto new_tokens = cortex_tokenizer_->Encode(role);
  tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
  tokens.push_back(
      static_cast<int32_t>(Llama3Template::kEndHeaderId));  // <|end_header_id|>
  tokens.push_back(static_cast<int32_t>(Llama3Template::kParagraph));  // \n\n
  return tokens;
}

std::vector<int> TensorrtllmEngine::EncodeMessageLlama3(
    const std::string& role, const std::string& content) {
  std::vector<int> tokens = EncodeHeaderLlama3(role);
  auto new_tokens = cortex_tokenizer_->Encode(content);
  tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
  tokens.push_back(
      static_cast<int32_t>(Llama3Template::kEndOfTurn));  // <|eot_id|>
  return tokens;
}
//#########################
//### ENGINE END POINTS ###
//#########################

void TensorrtllmEngine::HandleChatCompletion(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  inferences::ChatCompletionRequest request = inferences::fromJson(json_body);
  std::string formatted_input = pre_prompt_;
  nlohmann::json data;
  // data["stream"] = completion.stream;
  // data["n_predict"] = completion.max_tokens;
  data["presence_penalty"] = request.presence_penalty;
  Json::Value const& messages = request.messages;

  // tokens for Mistral v0.3
  // TODO(sang): too much hard code here, need to refactor it soon
  std::vector<int32_t> tokens;
  if (model_type_ == ModelType::kLlama3) {
    tokens.push_back(static_cast<int32_t>(
        Llama3Template::kBeginOfText));  // <|begin_of_text|>
  } else if (model_type_ == ModelType::kMistral) {
    tokens = {static_cast<int32_t>(MistralTemplate::kBos)};
  }
  // Format the input from user
  int msg_count = 0;
  for (auto const& message : messages) {
    std::string input_role = message["role"].asString();
    std::string role;
    if (model_type_ == ModelType::kLlama3) {
      std::string content = message["content"].asString();
      auto new_tokens = EncodeMessageLlama3(input_role, content);
      tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
    } else {
      if (input_role == "user") {
        role = user_prompt_;
        std::string content = message["content"].asString();
        formatted_input += role + content;
        if (model_type_ == ModelType::kMistral) {
          auto new_tokens = cortex_tokenizer_->Encode(content);
          new_tokens.insert(new_tokens.begin(),
                            static_cast<int32_t>(MistralTemplate::kBeginInst));
          new_tokens.push_back(static_cast<int32_t>(MistralTemplate::kEndInst));
          tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
        }
      } else if (input_role == "assistant") {
        role = ai_prompt_;
        std::string content = message["content"].asString();
        formatted_input += role + content;
        if (model_type_ == ModelType::kMistral) {
          auto new_tokens = cortex_tokenizer_->Encode(content);
          if (msg_count == messages.size() - 1) {
            new_tokens.push_back(static_cast<int32_t>(MistralTemplate::kEos));
          }
          tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
        }
      } else if (input_role == "system") {
        role = system_prompt_;
        std::string content = message["content"].asString();
        formatted_input = role + content + formatted_input;
      } else {
        role = input_role;
        std::string content = message["content"].asString();
        formatted_input += role + content;
      }
    }
    msg_count++;
  }
  formatted_input += ai_prompt_;

  std::vector<int32_t> input_ids_host;

  if (model_type_ == ModelType::kOpenHermes) {
    input_ids_host = cortex_tokenizer_->Encode(formatted_input);
  } else if (model_type_ == ModelType::kMistral) {
    input_ids_host = tokens;
  } else if (model_type_ == ModelType::kLlama3) {
    auto footer_tokens = EncodeHeaderLlama3("assistant");
    tokens.insert(tokens.end(), footer_tokens.begin(), footer_tokens.end());
    input_ids_host = tokens;
  }

  runtime_opts_.streaming = true;
  runtime_opts_.maxNewTokens = request.max_tokens;

  tle::OutputConfig outputConfig;
  outputConfig.excludeInputFromOutput = runtime_opts_.excludeInputFromOutput;
  tle::SamplingConfig samplingConfig(runtime_opts_.beamWidth);

  auto req_id = executor_->enqueueRequest(
      tle::Request(std::move(input_ids_host), runtime_opts_.maxNewTokens,
                   runtime_opts_.streaming, samplingConfig, outputConfig,
                   /*endId*/ std::nullopt, /*padId*/ std::nullopt,
                   /*badWords*/ std::nullopt, GetStopWords(model_type_)));

  q_->runTaskInQueue([this, cb = std::move(callback), req_id]() {
    auto& infer_state = responses_.Get(req_id);
    LOG_INFO << "Preparing to run inference task queue...";
    while (true) {  // Continuously check if the queue is not empty
      auto rew_text = infer_state.WaitAndPop();
      // std::cout << rew_text << std::endl;
      if (HandleMatch(rew_text, &infer_state, cb, model_type_) &&
          rew_text != "[DONE]") {
        continue;
      };
      // Simple bugfix for Mistral and Openhermes new line character
      if (rew_text == "<0x0A>")
        rew_text = "\n";

      if (rew_text == "[DONE]") {
        const std::string str = "data: " +
                                tensorrtllm_utils::CreateReturnJson(
                                    tensorrtllm_utils::GenerateRandomString(20),
                                    model_id_, "", "stop") +
                                "\n\n" + "data: [DONE]" + "\n\n";

        Json::Value resp_data;
        resp_data["data"] = str;
        Json::Value status;
        status["is_done"] = true;
        status["has_error"] = false;
        status["is_stream"] = true;
        status["status_code"] = k200OK;
        cb(std::move(status), std::move(resp_data));
        break;
      }
      const std::string text_to_stream =
          "data: " +
          tensorrtllm_utils::CreateReturnJson(
              tensorrtllm_utils::GenerateRandomString(20), model_id_,
              rew_text) +
          "\n\n";
      // std::cout << rew_text;

      Json::Value resp_data;
      resp_data["data"] = text_to_stream;
      Json::Value status;
      status["is_done"] = false;
      status["has_error"] = false;
      status["is_stream"] = true;
      status["status_code"] = k200OK;
      cb(std::move(status), std::move(resp_data));
    }
    // LOG_INFO << res_str;

    LOG_INFO << "Inference completed";
    responses_.Erase(req_id);
  });

  LOG_TRACE << "Done";
  return;
};

void TensorrtllmEngine::SetLoggerOption(const Json::Value& json_body) {
  if (!json_body["log_option"].isNull()) {
    int log_option = json_body["log_option"].asInt();
    if (log_option != kFileLoggerOption) {
      // Revert to default trantor logger output function
      trantor::Logger::setOutputFunction(
          [](const char* msg, const uint64_t len) {
            fwrite(msg, 1, static_cast<size_t>(len), stdout);
          },
          []() { fflush(stdout); });
    }
  }
  logger_ = std::make_shared<TllmFileLogger>();
  if (!json_body["log_level"].isNull()) {
    std::string log_level = json_body["log_level"].asString();
    if (log_level == "trace")
    {
      logger_->setLevel(nvinfer1::ILogger::Severity::kINFO);
      trantor::Logger::setLogLevel(trantor::Logger::kTrace);
    } else if (log_level == "debug") {
      trantor::Logger::setLogLevel(trantor::Logger::kDebug);
      logger_->setLevel(nvinfer1::ILogger::Severity::kINFO);
    } else if (log_level == "info") {
      trantor::Logger::setLogLevel(trantor::Logger::kInfo);
      logger_->setLevel(nvinfer1::ILogger::Severity::kINFO);
    } else if (log_level == "warn") {
      trantor::Logger::setLogLevel(trantor::Logger::kWarn);
      logger_->setLevel(nvinfer1::ILogger::Severity::kWARNING);
    } else if (log_level == "fatal") {
      trantor::Logger::setLogLevel(trantor::Logger::kFatal);
      logger_->setLevel(nvinfer1::ILogger::Severity::kWARNING);
    } else {
      trantor::Logger::setLogLevel(trantor::Logger::kError);
      logger_->setLevel(nvinfer1::ILogger::Severity::kERROR);
    }
  }
  else{
      logger_->setLevel(nvinfer1::ILogger::Severity::kWARNING);
  }
}

void TensorrtllmEngine::LoadModel(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  SetLoggerOption(*json_body);
  model::LoadModelRequest request = model::fromJson(json_body);
  if (model_loaded_ && model_type_ == GetModelType(request.model_path)) {
    LOG_INFO << "Model already loaded";
    Json::Value json_resp;
    json_resp["message"] = "Model already loaded";
    Json::Value status_resp;
    status_resp["status_code"] = k200OK;
    callback(std::move(status_resp), std::move(json_resp));
    return;
  } else {
    LOG_DEBUG << "Reset all resources and states before loading new model";
    Reset();
  }

  std::filesystem::path model_dir = request.model_path;
  model_type_ = GetModelType(request.model_path);
  n_parallel_ = request.n_parallel;
  batch_size_ = request.batch_size;
  LOG_DEBUG << "n_parallel: " << n_parallel_ << ", batch_size: " << batch_size_;

  int ctx_len = request.ctx_len;
  // We only support 3 models for now, it is ugly but it works :(
  if (model_type_ == ModelType::kOpenHermes) {
    user_prompt_ =
        request.user_prompt.empty() ? kOhUserPrompt : request.user_prompt;
    ai_prompt_ = request.ai_prompt.empty() ? kOhAiPrompt : request.ai_prompt;
    system_prompt_ =
        request.system_prompt.empty() ? kOhSystemPrompt : request.system_prompt;
  }
  model_id_ = GetModelId(*json_body);

  initTrtLlmPlugins(logger_.get());

  std::filesystem::path tokenizer_model_name = model_dir / "tokenizer.model";
  if (model_type_ == ModelType::kLlama3) {
    cortex_tokenizer_ =
        std::make_unique<TiktokenTokenizer>(tokenizer_model_name.string());
  } else {
    cortex_tokenizer_ =
        std::make_unique<SentencePieceTokenizer>(tokenizer_model_name.string());
  }

  LOG_INFO << "Loaded tokenizer from " << tokenizer_model_name.string();

  std::filesystem::path json_file_name = model_dir / "config.json";
  auto json = GptJsonConfig::parse(json_file_name);
  auto config = json.getModelConfig();
  auto world_config = WorldConfig::mpi(1, json.getTensorParallelism(),
                                       json.getPipelineParallelism());
  LOG_INFO << "Loaded config from " << json_file_name.string();

  auto model_path = model_dir / json.engineFilename(world_config, model_id_);

  runtime_opts_.beamWidth = 1;
  runtime_opts_.trtEnginePath = request.model_path;

  auto executor_config = tle::ExecutorConfig(
      runtime_opts_.beamWidth, tle::SchedulerConfig(), tle::KvCacheConfig(),
      /*enableChunkedContext*/ false, /*normalizeLogProbs*/ true,
      tle::kDefaultIterStatsMaxIterations,
      tle::kDefaultRequestStatsMaxIterations, tle::BatchingType::kINFLIGHT,
      batch_size_);
  try {
    executor_ = std::make_unique<tle::Executor>(runtime_opts_.trtEnginePath,
                                                tle::ModelType::kDECODER_ONLY,
                                                executor_config);
  } catch (const std::exception& e) {
    LOG_ERROR << "Failed to load model: " << e.what();
    executor_.reset();
    // Retry one more time
    try {
      executor_ = std::make_unique<tle::Executor>(runtime_opts_.trtEnginePath,
                                                  tle::ModelType::kDECODER_ONLY,
                                                  executor_config);
    } catch (const std::exception& e) {
      LOG_ERROR << "Failed to load model: " << e.what();
      executor_.reset();
      cortex_tokenizer_.reset();
      q_.reset();
      res_thread_.reset();
      logger_.reset();
      Json::Value json_resp;
      json_resp["message"] = "Failed to load model";
      Json::Value status;
      status["is_done"] = false;
      status["has_error"] = true;
      status["is_stream"] = false;
      status["status_code"] = k500InternalServerError;
      callback(std::move(status), std::move(json_resp));
      return;
    }
  }

  model_loaded_ = true;
  if (q_ == nullptr) {
    q_ = std::make_unique<trantor::ConcurrentTaskQueue>(n_parallel_, model_id_);
  }
  if (res_thread_ == nullptr) {
    res_thread_ = std::make_unique<std::thread>(
        &TensorrtllmEngine::WaitForResponses, this);
  }

  // Model loaded successfully
  LOG_INFO << "Model " << model_id_ << " loaded successfully from path "
           << model_path.string();
  Json::Value json_resp;
  json_resp["message"] = "Model loaded successfully";
  Json::Value status_resp;
  status_resp["status_code"] = k200OK;
  callback(std::move(status_resp), std::move(json_resp));
  start_time_ = std::chrono::system_clock::now().time_since_epoch() /
                std::chrono::milliseconds(1);
};

void TensorrtllmEngine::UnloadModel(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  if (!CheckModelLoaded(callback)) {
    LOG_WARN << "Model was not loaded";
    Json::Value json_resp;
    json_resp["message"] = "Model was not loaded";
    Json::Value status;
    status["status_code"] = k400BadRequest;
    callback(std::move(status), std::move(json_resp));
    return;
  }

  Reset();

  Json::Value json_resp;
  json_resp["message"] = "Model unloaded successfully";
  Json::Value status;
  status["is_done"] = true;
  status["has_error"] = false;
  status["is_stream"] = false;
  status["status_code"] = k200OK;
  callback(std::move(status), std::move(json_resp));
  LOG_INFO << "Model unloaded sucessfully";
}

void TensorrtllmEngine::HandleEmbedding(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  LOG_WARN << "Engine does not support embedding yet";
  Json::Value json_resp;
  json_resp["message"] = "Engine does not support embedding yet";
  Json::Value status;
  status["status_code"] = k409Conflict;
  callback(std::move(status), std::move(json_resp));
}

void TensorrtllmEngine::GetModelStatus(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  LOG_WARN << "Engine does not support get model status method yet";
  Json::Value json_resp;
  json_resp["message"] = "Engine does not support get model status method yet";
  Json::Value status;
  status["status_code"] = k409Conflict;
  callback(std::move(status), std::move(json_resp));
}

void TensorrtllmEngine::GetModels(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  Json::Value json_resp;
  Json::Value model_array = Json::arrayValue;

  if (model_loaded_) {
    Json::Value val;
    val["id"] = model_id_;
    val["engine"] = "cortex.tensorrt-llm";
    val["start_time"] = start_time_;
    val["vram"] = "-";
    val["ram"] = "-";
    val["object"] = "model";
    model_array.append(val);
  }

  json_resp["object"] = "list";
  json_resp["data"] = model_array;

  Json::Value status;
  status["is_done"] = true;
  status["has_error"] = false;
  status["is_stream"] = false;
  status["status_code"] = k200OK;
  callback(std::move(status), std::move(json_resp));
  LOG_INFO << "Running models responded";
}

bool TensorrtllmEngine::WaitForResponses() {
  // Get the new tokens for each request
  // TODO(sang) only works with beamWidth = 1 now
  while (model_loaded_) {
    std::chrono::milliseconds wait_time(10);
    // Wait for any response
    auto responses = executor_->awaitResponses(wait_time);
    // Loop over the responses
    for (auto const& response : responses) {
      // Map back to our request id
      auto request_id = response.getRequestId();

      if (!response.hasError()) {
        auto result = response.getResult();

        for (tle::SizeType32 beam = 0; beam < runtime_opts_.beamWidth; ++beam) {
          auto& resp_tokens = result.outputTokenIds.at(beam);
          responses_.Get(request_id).AddTokenGenCount(resp_tokens.size());
          RemoveSpecialTokens(resp_tokens, model_type_);
          if (resp_tokens.empty())
            continue;
          if (model_type_ == ModelType::kLlama3) {
            responses_.Get(request_id)
                .Enqueue(cortex_tokenizer_->Decode(resp_tokens));
          } else {
            for (auto res : resp_tokens) {
              responses_.Get(request_id)
                  .Enqueue(cortex_tokenizer_->DecodeWithSpace(res));
              // LOG_INFO << responses_[request_id].texts_to_stream.back();
            }
          }
        }
        if (result.isFinal) {
          LOG_INFO << "Request id " << request_id << " is completed.";
          responses_.Get(request_id).Enqueue("[DONE]");
        }
      } else {
        // Allow response with error only if awaitResponse processed a terminated request id
        std::string err = "ReqId " + std::to_string(response.getRequestId()) +
                          " has already been processed and was terminated.";
        if (response.getErrorMsg() != err) {
          LOG_ERROR << "Request id " << request_id << " encountered error "
                    << response.getErrorMsg().c_str();
          return false;
        }
      }
    }
  }

  return true;
}

void TensorrtllmEngine::Reset() {
  LOG_INFO << "Reset all resources and states";
  model_loaded_ = false;
  if (res_thread_ && res_thread_->joinable()) {
    res_thread_->join();
    res_thread_.reset();
  }
  executor_.reset();
  cortex_tokenizer_.reset();
  q_.reset();
  logger_.reset();
}

extern "C" {
EngineI* get_engine() {
  return new TensorrtllmEngine();
}
}
