#pragma once

#include "sentencepiece_processor.h"
#include <cstdint>
#include <drogon/HttpController.h>

#include "sentencepiece_processor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include <NvInfer.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

using namespace drogon;

using namespace tensorrt_llm::runtime;

class Tokenizer
{
private:
    sentencepiece::SentencePieceProcessor processor;

    void replaceSubstring(std::string& base, const std::string& from, const std::string& to)
    {
        size_t start_pos = 0;
        while ((start_pos = base.find(from, start_pos)) != std::string::npos)
        {
            base.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
    }

public:
    Tokenizer(const std::string& modelPath)
    {
        auto status = processor.Load(modelPath);
        if (!status.ok())
        {
            std::cerr << status.ToString() << std::endl;
        }
        LOG_INFO << "Successully loaded the tokenizer";
    }

    std::string decodeWithSpace(const int id)
    {
        std::string text = processor.IdToPiece(id);
        replaceSubstring(text, "▁", " ");
        return text;
    }

    std::string decode(const std::vector<int32_t> ids)
    {
        std::string text = processor.DecodeIds(ids);
        return text;
    }

    std::vector<int> encode(const std::string& input)
    {
        std::vector<int> ids;
        processor.Encode(input, &ids);
        return ids;
    }
};

class tensorrtllm : public drogon::HttpController<tensorrtllm>
{
public:
    tensorrtllm()
    {
        std::vector<int> text_input = nitro_tokenizer.encode(example_string);
        const int inputLen = text_input.size();
        const std::vector<int> inOutLen = {inputLen, 2000}; // input_length, output_length

        logger = std::make_shared<TllmLogger>();
        logger->setLevel(nvinfer1::ILogger::Severity::kINFO);
        // Fixed settings
        const std::string modelName = "mistral";
        const std::filesystem::path engineDir = "/app/mistral_engine_3/";
        const int batchSize = 1;
        initTrtLlmPlugins(logger.get());
        // Load model configuration
        std::filesystem::path jsonFileName = engineDir / "config.json";
        auto const json = GptJsonConfig::parse(jsonFileName);
        auto config = json.getModelConfig();
        modelConfig = std::make_unique<GptModelConfig>(config);
        auto const worldConfig = WorldConfig::mpi(1, json.getTensorParallelism(), json.getPipelineParallelism());
        auto const enginePath = engineDir / json.engineFilename(worldConfig, modelName);
        auto const dtype = modelConfig->getDataType();

        // Set gptsessionconfig
        sessionConfig.maxBatchSize = batchSize;
        sessionConfig.maxBeamWidth = 1; // Fixed for simplicity
        sessionConfig.maxSequenceLength = inOutLen[0] + inOutLen[1];
        sessionConfig.cudaGraphMode = false; // Fixed for simplicity

        // Set smapling config
        samplingConfig.temperature = std::vector{0.0f};
        samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
        samplingConfig.topK = std::vector{40};
        samplingConfig.topP = std::vector{0.0f};
        samplingConfig.minLength = std::vector{inOutLen[1]};
        samplingConfig.repetitionPenalty = std::vector{1.3f};
        gptSession
            = std::make_unique<GptSession>(sessionConfig, *modelConfig, worldConfig, enginePath.string(), logger);
    };

    METHOD_LIST_BEGIN
    // use METHOD_ADD to add your custom processing function here;
    // METHOD_ADD(tensorrtllm::get, "/{2}/{1}", Get); // path is /tensorrtllm/{arg2}/{arg1}
    // METHOD_ADD(tensorrtllm::your_method_name, "/{1}/{2}/list", Get); // path is /tensorrtllm/{arg1}/{arg2}/list
    ADD_METHOD_TO(tensorrtllm::chat_completion, "/testing", Get); // path is
    // /absolute/path/{arg1}/{arg2}/list

    METHOD_LIST_END
    // your declaration of processing function maybe like this:
    // void get(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback, int p1, std::string
    // p2);
    void chat_completion(const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback);

private:
    GptSession::Config sessionConfig{1, 1, 1};
    SamplingConfig samplingConfig{1};
    std::unique_ptr<GptModelConfig> modelConfig;
    Tokenizer nitro_tokenizer{"./new_chatml_tokenizer.model"};
    std::unique_ptr<GptSession> gptSession;
    std::shared_ptr<TllmLogger> logger;
    std::string example_string{
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant"};
};
