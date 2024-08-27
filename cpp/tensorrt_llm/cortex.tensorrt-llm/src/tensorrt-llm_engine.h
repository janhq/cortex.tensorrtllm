#pragma once

#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>

#include <nlohmann/json.hpp>
#include "NvInfer.h"
#include "base/cortex-common/enginei.h"
#include "cpp-tiktoken/emdedded_resource_reader.h"  //include to use tiktoken
#include "cpp-tiktoken/encoding.h"                  //include to use tiktoken
#include "models/chat_completion_request.h"
#include "models/load_model_request.h"
#include "sentencepiece_processor.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"
#include <trantor/utils/AsyncFileLogger.h>

using namespace tensorrt_llm::runtime;

namespace tle = tensorrt_llm::executor;

namespace fs = std::filesystem;

namespace tc = tensorrt_llm::common;

constexpr char log_base_name[] = "logs/cortex";
constexpr char log_folder[] = "logs";
constexpr size_t max_log_file_size = 20000000; // ~20mb

// This class is inspired by https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/tllmLogger.cpp
class TllmFileLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity,
           nvinfer1::AsciiChar const* msg) noexcept override {
    switch (severity) {
      case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        LOG_ERROR << "[TensorRT-LLM][ERROR] " << msg;
        break;
      case nvinfer1::ILogger::Severity::kERROR:
        LOG_ERROR << "[TensorRT-LLM][ERROR] " << msg;
        break;
      case nvinfer1::ILogger::Severity::kWARNING:
        LOG_WARN << "[TensorRT-LLM][WARN] " << msg;
        break;
      case nvinfer1::ILogger::Severity::kINFO:
        LOG_INFO << "[TensorRT-LLM][INFO] " << msg;
        break;
      case nvinfer1::ILogger::Severity::kVERBOSE:
        LOG_DEBUG << "[TensorRT-LLM][DEBUG] " << msg;
        break;
      default:
        LOG_TRACE << "[TensorRT-LLM][TRACE] " << msg;
        break;
    }
  }
  Severity getLevel() {
    auto* const logger = tc::Logger::getLogger();
    switch (logger->getLevel())
    {
    case tc::Logger::Level::ERROR: return nvinfer1::ILogger::Severity::kERROR;
    case tc::Logger::Level::WARNING: return nvinfer1::ILogger::Severity::kWARNING;
    case tc::Logger::Level::INFO: return nvinfer1::ILogger::Severity::kINFO;
    case tc::Logger::Level::DEBUG:
    case tc::Logger::Level::TRACE: return nvinfer1::ILogger::Severity::kVERBOSE;
    default: return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    }
  };

  void setLevel(Severity level) {
    auto* const logger = tc::Logger::getLogger();
    switch (level) {
      case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        logger->setLevel(tc::Logger::Level::ERROR);
        break;
      case nvinfer1::ILogger::Severity::kERROR:
        logger->setLevel(tc::Logger::Level::ERROR);
        break;
      case nvinfer1::ILogger::Severity::kWARNING:
        logger->setLevel(tc::Logger::Level::WARNING);
        break;
      case nvinfer1::ILogger::Severity::kINFO:
        logger->setLevel(tc::Logger::Level::INFO);
        break;
      case nvinfer1::ILogger::Severity::kVERBOSE:
        logger->setLevel(tc::Logger::Level::TRACE);
        break;
      default:
        TLLM_THROW("Unsupported severity");
    }
  };
};

struct RuntimeOptions {
  std::string trtEnginePath;

  bool streaming;
  bool excludeInputFromOutput = true;
  tle::SizeType32 maxNewTokens;
  tle::SizeType32 beamWidth;
  tle::SizeType32 timeoutMs;

  bool useOrchestratorMode;
  std::string workerExecutablePath;
};

// This class is file source reader from https://github.com/gh-markt/cpp-tiktoken/blob/master/ut/tests.cpp
class TFilePathResourceReader : public IResourceReader {
 public:
  TFilePathResourceReader(const std::string& path) : path_(path) {}

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
  Tokenizer() {}

  virtual std::string DecodeWithSpace(const int id) { return ""; }

  virtual std::string Decode(const std::vector<int32_t> ids) = 0;

  virtual std::vector<int> Encode(const std::string& input) = 0;
};

class SentencePieceTokenizer : public Tokenizer {
 private:
  sentencepiece::SentencePieceProcessor processor;

  void ReplaceSubstring(std::string& base, const std::string& from,
                        const std::string& to) {
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
    std::lock_guard<std::mutex> l(m_);
    std::string text = processor.IdToPiece(id);
    ReplaceSubstring(text, "▁", " ");
    return text;
  }

  std::string Decode(const std::vector<int32_t> ids) override {
    std::lock_guard<std::mutex> l(m_);
    std::string text = processor.DecodeIds(ids);
    return text;
  }

  std::vector<int> Encode(const std::string& input) override {
    std::lock_guard<std::mutex> l(m_);
    std::vector<int> ids;
    processor.Encode(input, &ids);
    return ids;
  }

 private:
  std::mutex m_;
};

class TiktokenTokenizer : public Tokenizer {
 private:
  std::shared_ptr<GptEncoding> encoder;

 public:
  TiktokenTokenizer(const std::string& model_path) : Tokenizer() {
    TFilePathResourceReader reader(model_path);
    encoder =
        GptEncoding::get_encoding_llama3(LanguageModel::CL100K_BASE, &reader);
    LOG_INFO << "Successully loaded the tokenizer";
  }

  std::string Decode(const std::vector<int32_t> ids) override {
    std::lock_guard<std::mutex> l(m_);
    std::string text = encoder->decode(ids);
    return text;
  }

  std::vector<int> Encode(const std::string& input) override {
    std::lock_guard<std::mutex> l(m_);
    std::vector<int> ids = encoder->encode(input);
    return ids;
  }

 private:
  std::mutex m_;
};
enum class ModelType { kOpenHermes, kLlama3, kMistral };

struct InferenceState {
  void Reset() { stop_word_match_len = 0; }

  bool IsComplete(ModelType model_type) const {
    if (model_type == ModelType::kOpenHermes ||
        model_type == ModelType::kLlama3) {
      return stop_word_match_len >= sequence_openhermes.size();
    } else {
      return stop_word_match_len >= sequence_mistral.size();
    }
  }

  const std::string& GetSequence(ModelType model_type, size_t index) {
    if (model_type == ModelType::kOpenHermes ||
        model_type == ModelType::kLlama3) {
      return sequence_openhermes[index];
    } else {
      return sequence_mistral[index];
    }
  }

  void Enqueue(std::string s) {
    std::lock_guard<std::mutex> l(m);
    texts_to_stream.push(std::move(s));
    cv.notify_one();
  }

  std::string WaitAndPop() {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [this]() { return !texts_to_stream.empty(); });
    auto s = texts_to_stream.front();
    texts_to_stream.pop();
    return s;
  }

  size_t GetStopMatchLen() const {
    std::lock_guard<std::mutex> l(m);
    return stop_word_match_len;
  }

  void AddStopMatch() {
    std::lock_guard<std::mutex> l(m);
    stop_word_match_len++;
  }

  void ResetStopMatch() {
    std::lock_guard<std::mutex> l(m);
    stop_word_match_len = 1;
  }

  void AddTokenGenCount(int count) {
    std::lock_guard<std::mutex> l(m);
    token_gen_count += count;
  }

 private:
  std::queue<std::string> texts_to_stream;
  mutable std::mutex m;  // Mutex to protect access to texts_to_stream
  std::condition_variable cv;
  size_t stop_word_match_len = 0;
  std::vector<std::string> sequence_openhermes = {"<",   "|", "im", "_",
                                                  "end", "|", ">"};
  std::vector<std::string> sequence_mistral = {"</s>"};
  int token_gen_count = 0;
};

namespace tensorrtllm {

class TensorrtllmEngine : public EngineI {
 public:
  TensorrtllmEngine(int log_option = 0);
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
  virtual std::vector<int> EncodeMessageLlama3(const std::string& role,
                                               const std::string& content);
  // API to get running models.
  void GetModels(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void SetFileLogger();
 private:
  bool CheckModelLoaded(
      std::function<void(Json::Value&&, Json::Value&&)>& callback);

  // Function that waits for responses and stores output tokens
  bool WaitForResponses();

  // Release all resources and states
  void Reset();

 private:
  std::unique_ptr<std::thread>
      res_thread_;  // worker thread to handle responses
  template <typename T>
  struct InfSyncMap {
    T& Get(uint64_t k) {
      std::lock_guard<std::mutex> l(m);
      return data[k];
    }

    void Erase(uint64_t k) {
      std::lock_guard<std::mutex> l(m);
      if (data.find(k) != data.end())
        data.erase(k);
    }

   private:
    std::mutex m;
    std::unordered_map<uint64_t, T> data;
  };
  InfSyncMap<InferenceState> responses_;

  std::unique_ptr<Tokenizer> cortex_tokenizer_;
  RuntimeOptions runtime_opts_;
  std::unique_ptr<tle::Executor> executor_;
  std::shared_ptr<TllmFileLogger> logger_;
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
  int n_parallel_ = 1;
  std::unique_ptr<trantor::AsyncFileLogger> asynce_file_logger_;
};

}  // namespace tensorrtllm
