#pragma once

#include "base/cortex-common/enginei.h"
#include "models/chat_completion_request.h"
#include "models/load_model_request.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"
#include "sentencepiece_processor.h"
#include <queue>

using namespace tensorrt_llm::runtime;

class Tokenizer
{
 private:
  sentencepiece::SentencePieceProcessor processor;

  void ReplaceSubstring(std::string& base, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = base.find(from, start_pos)) != std::string::npos) {
      base.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
  }

 public:
  Tokenizer(const std::string& model_path) {
    auto status = processor.Load(model_path);
    if (!status.ok()) {
      LOG_ERROR << status.ToString();
      }
    LOG_INFO << "Successully loaded the tokenizer";
  }

  std::string DecodeWithSpace(const int id) {
    std::string text = processor.IdToPiece(id);
    ReplaceSubstring(text, "▁", " ");
    return text;
  }

  std::string Decode(const std::vector<int32_t> ids) {
    std::string text = processor.DecodeIds(ids);
    return text;
  }

  std::vector<int> Encode(const std::string& input) {
    std::vector<int> ids;
    processor.Encode(input, &ids);
    return ids;
  }
};

struct InferenceState {
  int prev_pos{0};
  std::string prev_text;
  bool is_finished;
  std::queue<std::string> texts_to_stream;
  std::mutex queue_mutex; // Mutex to protect access to textsToStream
  size_t stop_word_match_len = 0;
  std::vector<std::string> sequence{"<", "|", "im", "_", "end", "|", ">"};

  void reset() {
    stop_word_match_len = 0;
    prev_text = "";
  }

  bool isComplete() const {
    return stop_word_match_len >= sequence.size();
  }
};

class TensorrtllmEngine : public EngineI {
  public:
    TensorrtllmEngine();
    ~TensorrtllmEngine() final;
    // ### Interface ###
  void LoadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void HandleChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void Destroy(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;  
  GenerationInput CreateGenerationInput(std::vector<int32_t> inputIds);
  GenerationOutput CreateGenerationOutput();
  GenerationInput::TensorPtr GetTensorChatMLStopWordList();

  std::unique_ptr<Tokenizer> cortex_tokenizer;
  std::unique_ptr<GptSession> gpt_session;

 private:
  bool LoadModelImpl(
      tensorrtllm::model::LoadModelRequest&& request,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  void HandleInferenceImpl(
      tensorrtllm::inferences::ChatCompletionRequest&& request,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  void DestroyImpl(
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);  
  void GetFormattedInputFromMessages(
      const Json::Value& messages,
      std::string& formatted_input);

  int batch_size = 1;
  std::string ai_prompt;
  std::string user_prompt;
  std::string system_prompt;
  std::string pre_prompt;
  std::shared_ptr<TllmLogger> logger;
  std::unique_ptr<ModelConfig> model_config;
  GptSession::Config session_config{1, 1, 1};

 private:
  struct ServerInfo {
    std::unique_ptr<trantor::ConcurrentTaskQueue> q;
    // std::string user_prompt;
    // std::string ai_prompt;
    // std::string system_prompt;
    // std::string pre_prompt;
    // int repeat_last_n;
    // bool caching_enabled;
    // std::string grammar_file_content;
  };

  // key: model_id, value: ServerInfo
  std::unordered_map<std::string, ServerInfo> server_map_;  
};
