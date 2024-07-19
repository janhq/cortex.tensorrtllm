#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>

#include "NvInfer.h"
#include "base/cortex-common/enginei.h"
#include "models/chat_completion_request.h"
#include "models/load_model_request.h"
#include "sentencepiece_processor.h"
#include "cpp-tiktoken/encoding.h" //include to use tiktoken
#include "cpp-tiktoken/emdedded_resource_reader.h" //include to use tiktoken
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"
#include <nlohmann/json.hpp>


using namespace tensorrt_llm::runtime;
// This class is file source reader from https://github.com/gh-markt/cpp-tiktoken/blob/master/ut/tests.cpp
class TFilePathResourceReader : public IResourceReader {
public:
    TFilePathResourceReader(const std::string& path) 
        : path_(path)
    {
    }

    std::vector<std::string> readLines() override {
        std::ifstream file(path_);
        if (!file.is_open()) {
            throw std::runtime_error("Embedded resource '" + path_ + "' not found.");
        }

        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        return lines;
    }
private:
    std::string path_;
};

class Tokenizer {

 public:
  Tokenizer() {
  }

  virtual std::string DecodeWithSpace(const int id) {
  }

  virtual std::string Decode(const std::vector<int32_t> ids) {
  }

  virtual std::vector<int> Encode(const std::string& input) {
  }
};

class SentencePieceTokenizer : public Tokenizer {
 private:
  sentencepiece::SentencePieceProcessor processor;

  void ReplaceSubstring(std::string& base, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = base.find(from, start_pos)) != std::string::npos) {
        base.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
  }

 public:
  SentencePieceTokenizer(const std::string& model_path) : Tokenizer() {
    auto status = processor.Load(model_path);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
    }
    LOG_INFO << "Successully loaded the tokenizer";
  }

  std::string DecodeWithSpace(const int id) override {
    std::string text = processor.IdToPiece(id);
    ReplaceSubstring(text, "▁", " ");
    return text;
  }

  std::string Decode(const std::vector<int32_t> ids) override {
    std::string text = processor.DecodeIds(ids);
    return text;
  }

  std::vector<int> Encode(const std::string& input) override {
    std::vector<int> ids;
    processor.Encode(input, &ids);
    return ids;
  }
};

class TiktokenTokenizer : public Tokenizer {
 private:
  std::shared_ptr<GptEncoding> encoder;

 public:
  TiktokenTokenizer(const std::string& model_path) : Tokenizer() {
    TFilePathResourceReader reader(model_path);
    encoder = GptEncoding::get_encoding_llama3(LanguageModel::CL100K_BASE, &reader);
    LOG_INFO << "Successully loaded the tokenizer";
  }

  std::string Decode(const std::vector<int32_t> ids) override {
    std::string text = encoder->decode(ids);
    return text;
  }

  std::vector<int> Encode(const std::string& input) override {
    std::vector<int> ids = encoder->encode(input);
    return ids;
  }
};
  enum class ModelType {
    kOpenHermes, kLlama3, kMistral
};

struct InferenceState {
  int prev_pos{0};
  bool is_finished;
  std::queue<std::string> texts_to_stream;
  std::mutex queue_mutex; // Mutex to protect access to textsToStream
  size_t stop_word_match_len = 0;
  std::vector<std::string> sequence_openhermes = {"<", "|", "im", "_", "end", "|", ">"};
  std::vector<std::string> sequence_mistral = {"[", "INST", "]"};
  int token_gen_count = 0;

  void Reset() {
    stop_word_match_len = 0;
  }

  bool IsComplete(ModelType model_type) const {
    if(model_type == ModelType::kOpenHermes || model_type == ModelType::kLlama3) {
      return stop_word_match_len >= sequence_openhermes.size();
    } else {
      return stop_word_match_len >= sequence_mistral.size();
    }
  }

  const std::string& GetSequence(ModelType model_type, size_t index) {
    if(model_type == ModelType::kOpenHermes || model_type == ModelType::kLlama3) {
      return sequence_openhermes[index];
    } else {
      return sequence_mistral[index];
    }

  }
};

namespace tensorrtllm {


class TensorrtllmEngine : public EngineI {
 public:
  ~TensorrtllmEngine() final;
  // ### Interface ###
  void HandleChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void HandleEmbedding(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void LoadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void UnloadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void GetModelStatus(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  virtual std::vector<int> EncodeHeaderLlama3(const std::string& role);
  virtual std::vector<int> EncodeMessageLlama3( const std::string& role, const std::string& content);
  // API to get running models.
  void GetModels(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;

  GenerationInput::TensorPtr GetTensorSingleStopWordList(int stopToken);
  GenerationInput CreateGenerationInput(std::vector<int32_t> inputIds);
  GenerationOutput CreateGenerationOutput();
  GenerationInput::TensorPtr GetTensorChatMLStopWordList();

  std::unique_ptr<GptSession> gpt_session;
  std::unique_ptr<Tokenizer> cortex_tokenizer;

 private:
  bool CheckModelLoaded(
      std::function<void(Json::Value&&, Json::Value&&)>& callback);

  GptSession::Config session_config_{1, 1, 1};
  std::unique_ptr<ModelConfig> model_config_;
  std::shared_ptr<TllmLogger> logger_;
  std::string user_prompt_;
  std::string ai_prompt_;
  std::string system_prompt_;
  std::string pre_prompt_;
  int batch_size_ = 1;
  std::string model_id_;
  uint64_t start_time_;
  std::atomic<bool> model_loaded_;
  std::unique_ptr<trantor::ConcurrentTaskQueue> q_;
  ModelType model_type_ = ModelType::kOpenHermes;
};

} // namespace inferences
