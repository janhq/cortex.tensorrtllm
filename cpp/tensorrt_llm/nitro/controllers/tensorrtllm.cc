#include "tensorrtllm.h"
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <trantor/utils/Logger.h>
#include <vector>

void removeZeroes(std::vector<int>& vec)
{
    vec.erase(std::remove(vec.begin(), vec.end(), 0), vec.end());
}

struct inferenceState
{
    int prevPos{0};
    bool isFinished;
};

void tensorrtllm::chat_completion(const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback)
{
    std::shared_ptr<inferenceState> inferState = std::make_shared<inferenceState>();

    std::vector<int> text_input = nitro_tokenizer.encode(example_string);
    const int inputLen = text_input.size();
    const std::vector<int> inOutLen = {inputLen, 2000}; // input_length, output_length

    const int batchSize = 1;

    std::vector<int32_t> inputIdsHost = text_input;

    std::cout << "Start Nitro testing session: " << std::endl;

    auto& bufferManager = gptSession->getBufferManager();
    // Make stopwordlists

    // Your stop word single token "32000"
    std::vector<int32_t> stopWordsTokens = {32000, -1, 1, -1}; // Extend with -1 for increased length

    // Tensor creation for stopWordsList
    // Assuming the framework allows similar operations for creating custom tensors
    // At this point,
    // Input preparation
    GenerationInput::TensorPtr inputIds
        = bufferManager.copyFrom(inputIdsHost, ITensor::makeShape({batchSize, inOutLen[0]}), MemoryType::kGPU);

    std::vector<int32_t> inputLengthsHost(batchSize, inOutLen[0]);
    GenerationInput::TensorPtr inputLengths
        = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);
    bool inputPacked = modelConfig->usePackedInput();

    GenerationInput generationInput{0, 32000, inputIds, inputLengths, inputPacked};

    // generationInput.stopWordsList = stopWordsTokensTensor;

    generationInput.stopWordsList = bufferManager.copyFrom(stopWordsTokens, ITensor::makeShape({2,2}), MemoryType::kGPU);
    generationInput.stopWordsList->reshape(ITensor::makeShape({1,2,2}));

    LOG_INFO << "here is the shape:   " << generationInput.stopWordsList->getShape().d[0];

    LOG_INFO << "here is the shape:   " << generationInput.stopWordsList->getShape().d[1];
    GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

    // Define the callback to stream each generated token
    generationOutput.onTokenGenerated = [&inferState, &bufferManager, inOutLen, this, &generationOutput](
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
            std::vector<int> outputIdsHostDecode(outputIdsHost.begin() + inOutLen[0], outputIdsHost.end());
            removeZeroes(outputIdsHostDecode);
            std::string text = nitro_tokenizer.decode(outputIdsHostDecode);

            if (inferState->prevPos > 0 && inferState->prevPos < text.size())
            {
                // Valid prevPos, proceed with slicing the string from prevPos to the end
                std::string stringTok(text.begin() + inferState->prevPos, text.end());
                std::cout << stringTok << std::flush;
            }
            else if (inferState->prevPos >= text.size())
            {
                // prevPos is out of bounds, indicating there might be no new text or an error in logic
                // You can handle this case as needed, for example, by logging a warning
                inferState->prevPos = text.size();
            }
            else
            {
                // inferState->prevPos is 0 or negative, indicating a potential logic error or initial state
                // If there's valid text, you might want to print it all or handle this case specifically
                if (!text.empty())
                {
                    std::cout << text << std::flush; // Optionally print all text if it's the initial state
                }
            }
            // Update prevPos to the new length of the text for the next iteration
            inferState->prevPos = text.size();
        }
    };

    gptSession->generate(generationOutput, generationInput, samplingConfig);

    bufferManager.getStream().synchronize();

    auto resp=HttpResponse::newHttpResponse();
    resp->setStatusCode(k200OK);
    resp->setBody("Your Page Contents");
    callback(resp);
    LOG_INFO << "Hello world";
    return;
};

// Add definition of your processing function here
