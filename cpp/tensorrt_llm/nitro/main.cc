#include "sentencepiece_processor.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "thread"
#include <NvInfer.h>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace trt = nvinfer1;

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
    }

    std::string decodeWithSpace(const int id)
    {
        std::string text = processor.IdToPiece(id);
        replaceSubstring(text, "▁", " ");
        return text;
    }

    std::vector<int> encode(const std::string& input)
    {
        std::vector<int> ids;
        processor.Encode(input, &ids);
        return ids;
    }
};

namespace
{
void runBenchmark()
{
    Tokenizer nitro_tokenizer("./tokenizer.model");
    std::vector<int> text_input = nitro_tokenizer.encode("How to survive in the Abyss chapter 1:\n\n ");

    // Fixed settings
    const std::string modelName = "mistral";
    const std::filesystem::path engineDir = "/app/mistral_engine_2/";
    const int batchSize = 1;
    const int inputLen = text_input.size();
    const std::vector<int> inOutLen = {inputLen, 500}; // input_length, output_length

    // Logger setup
    auto logger = std::make_shared<TllmLogger>();
    logger->setLevel(nvinfer1::ILogger::Severity::kINFO);

    initTrtLlmPlugins(logger.get());

    // Load model configuration
    std::filesystem::path jsonFileName = engineDir / "config.json";
    auto const json = GptJsonConfig::parse(jsonFileName);
    auto const modelConfig = json.getModelConfig();
    auto const worldConfig = WorldConfig::mpi(1, json.getTensorParallelism(), json.getPipelineParallelism());
    auto const enginePath = engineDir / json.engineFilename(worldConfig, modelName);
    auto const dtype = modelConfig.getDataType();

    GptSession::Config sessionConfig{1, 1, 1};
    sessionConfig.maxBatchSize = batchSize;
    sessionConfig.maxBeamWidth = 4; // Fixed for simplicity
    sessionConfig.maxSequenceLength = inOutLen[0] + inOutLen[1];
    sessionConfig.cudaGraphMode = false; // Fixed for simplicity

    SamplingConfig samplingConfig{1}; // Fixed for simplicity
    samplingConfig.temperature = std::vector{0.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{40};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{inOutLen[1]};
    samplingConfig.repetitionPenalty = std::vector{1.3f};

    // Initialize session
    GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};
    // Generate random input IDs within the model's vocabulary range
    const int vocabSize = modelConfig.getVocabSize();
    std::vector<int32_t> inputIdsHost = text_input;

    std::cout << "Start Nitro testing session: " << std::endl;
    // Input preparation
    auto& bufferManager = session.getBufferManager();
    GenerationInput::TensorPtr inputIds
        = bufferManager.copyFrom(inputIdsHost, ITensor::makeShape({batchSize, inOutLen[0]}), MemoryType::kGPU);

    std::vector<int32_t> inputLengthsHost(batchSize, inOutLen[0]);
    GenerationInput::TensorPtr inputLengths
        = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    bool inputPacked = modelConfig.usePackedInput();

    GenerationInput generationInput{0, 0, inputIds, inputLengths, inputPacked};

    GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};
    // Define the callback to stream each generated token
    generationOutput.onTokenGenerated = [&bufferManager, inOutLen, &nitro_tokenizer, &generationOutput](
                                            GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished)
    {
        if (!finished)
        {
            // Assuming the shape of outputIds tensor is (1, 1, 160), where 160 is the number of tokens
            int outputLength = outputIds->getShape().d[2]; // Get the length of output IDs based on the tensor shape
            // Copy output IDs from GPU to host for printing
            std::vector<int32_t> outputIdsHost(outputLength);
            bufferManager.copy(*outputIds, outputIdsHost.data(), MemoryType::kCPU);
            // Find the last non-zero value in the output IDs starting from the end of the input sequence
            int lastNonZeroIndex = -1;
            for (int i = outputLength - 1; i >= inOutLen[0]; --i)
            {
                if (outputIdsHost[i] != 0)
                {
                    lastNonZeroIndex = i;
                    break; // Stop at the first non-zero token found from the end
                }
            }

            // Directly print the last non-zero value if found, without using 'step'
            if (lastNonZeroIndex != -1)
            {
                int outTok = outputIdsHost[lastNonZeroIndex];
                if (outTok == 13)
                {
                    std::cout << "\n";
                }
                else
                {
                    std::cout << nitro_tokenizer.decodeWithSpace(outTok);
                }
            }
        }
    };

    session.generate(generationOutput, generationInput, samplingConfig);
    bufferManager.getStream().synchronize();
}

} // namespace

int main()
{
    try
    {
        runBenchmark();
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
