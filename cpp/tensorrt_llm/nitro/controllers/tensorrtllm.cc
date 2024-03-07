#include "tensorrtllm.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "utils/nitro_utils.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>
#include <trantor/utils/Logger.h>
#include <vector>

void removeId(std::vector<int>& vec, int id)
{
    vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
}

struct inferenceState
{
    int prevPos{0};
    bool isFinished;
    std::queue<std::string> textsToStream;
    std::mutex queueMutex; // Mutex to protect access to textsToStream
};

// Only support single token stopping point now
std::string create_return_json(const std::string& id, const std::string& model, const std::string& content,
    Json::Value finish_reason = Json::Value())
{
    Json::Value root;

    root["id"] = id;
    root["model"] = model;
    root["created"] = static_cast<int>(std::time(nullptr));
    root["object"] = "chat.completion.chunk";

    Json::Value choicesArray(Json::arrayValue);
    Json::Value choice;

    choice["index"] = 0;
    Json::Value delta;
    delta["content"] = content;
    choice["delta"] = delta;
    choice["finish_reason"] = finish_reason;

    choicesArray.append(choice);
    root["choices"] = choicesArray;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = ""; // This sets the indentation to an empty string,
                                // producing compact output.
    return Json::writeString(writer, root);
}

GenerationInput::TensorPtr tensorrtllm::getTensorSingleStopWordList(int stopToken)
{

    std::vector<int32_t> stopWordsTokens = {stopToken, -1, 1, -1}; // Extend with -1 for increased length
    return gptSession->getBufferManager().copyFrom(stopWordsTokens, ITensor::makeShape({1, 2, 2}), MemoryType::kGPU);
}

GenerationInput tensorrtllm::createGenerationInput(std::vector<int32_t> inputIdsHost)
{
    int inputLen = inputIdsHost.size();
    std::vector<int32_t> inputLengthsHost(batchSize, inputLen);
    GenerationInput::TensorPtr inputLengths
        = gptSession->getBufferManager().copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);
    GenerationInput::TensorPtr inputIds = gptSession->getBufferManager().copyFrom(
        inputIdsHost, ITensor::makeShape({batchSize, inputLen}), MemoryType::kGPU);

    GenerationInput generationInput{0, 0, inputIds, inputLengths, modelConfig->usePackedInput()};

    generationInput.stopWordsList = getTensorSingleStopWordList(32000);
    return generationInput;
}

GenerationOutput tensorrtllm::createGenerationOutput()
{
    GenerationOutput generationOutput{
        gptSession->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        gptSession->getBufferManager().emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};
    return generationOutput;
}


void inferenceThread(std::shared_ptr<inferenceState> inferState, 
                     std::vector<int32_t> inputIdsHost, 
                     std::function<void(const HttpResponsePtr&)> callback,
                     tensorrtllm* self)
{
    const int inputLen = inputIdsHost.size();
    const int outputLen = 2048 - inputLen;

    // Create sampling config
    SamplingConfig samplingConfig{1};
    samplingConfig.temperature = std::vector{0.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{40};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{outputLen};
    samplingConfig.repetitionPenalty = std::vector{1.3f};

    std::cout << "Start Nitro testing session: " << std::endl;

    // Input preparation

    GenerationInput generationInput = self->createGenerationInput(inputIdsHost);

    GenerationOutput generationOutput = self->createGenerationOutput();


        // Define the callback to stream each generated token
    generationOutput.onTokenGenerated = [&inferState, inputLen, outputLen, self, &generationOutput](
                                            GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished)
    {
        if (!finished)
        {
            // Assuming the shape of outputIds tensor is (1, 1, 160), where 160 is the number of tokens
            int outputLength = outputIds->getShape().d[2]; // Get the length of output IDs based on the tensor shape
            // Copy output IDs from GPU to host for printing
            std::vector<int32_t> outputIdsHost(outputLength);
            self->gptSession->getBufferManager().copy(*outputIds, outputIdsHost.data(), MemoryType::kCPU);
            // Find the last non-zero value in the output IDs starting from the end of the input sequence
            std::vector<int> outputIdsHostDecode(outputIdsHost.begin() + inputLen, outputIdsHost.end());
            removeId(outputIdsHostDecode, 0);
            removeId(outputIdsHostDecode, 32000);
            std::string text = self->nitro_tokenizer.decode(outputIdsHostDecode);

            if (inferState->prevPos > 0 && inferState->prevPos < text.size())
            {
                // Valid prevPos, proceed with slicing the string from prevPos to the end
                std::string stringTok(text.begin() + inferState->prevPos, text.end());
                std::lock_guard<std::mutex> guard(inferState->queueMutex); // Protect access with a lock
                inferState->textsToStream.push(stringTok);
            }
            else if (inferState->prevPos >= text.size())
            {
                inferState->prevPos = text.size();
            }
            inferState->prevPos = text.size();
        }
    };
    // The rest of the logic inside the `chat_completion` remains unchanged...
    // After finishing the setup, call the inference logic
    self->gptSession->generate(generationOutput, generationInput, samplingConfig);
}


void tensorrtllm::chat_completion(const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback)
{
    std::shared_ptr<inferenceState> inferState = std::make_shared<inferenceState>();

    std::vector<int32_t> inputIdsHost = nitro_tokenizer.encode(example_string);
    const int inputLen = inputIdsHost.size();
    const int outputLen = 2048 - inputLen;

    // Create sampling config
    SamplingConfig samplingConfig{1};
    samplingConfig.temperature = std::vector{0.0f};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{40};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.minLength = std::vector{outputLen};
    samplingConfig.repetitionPenalty = std::vector{1.3f};

    std::cout << "Start Nitro testing session: " << std::endl;

    // Input preparation

    std::thread infThread(inferenceThread, inferState, inputIdsHost, callback, this);
    infThread.detach(); // Detach the thread to allow it to run independently


    auto chunked_content_provider = [inferState](char* pBuffer, std::size_t nBuffSize) -> std::size_t
    {
            std::cout << "EMPTY";
        if (!pBuffer)
        {
            LOG_INFO << "Connection closed or buffer is null. Reset context";
            return 0; // Indicate no more data to send
        }

        while (true) // Continuously check if the queue is not empty
        {
            std::unique_lock<std::mutex> lock(inferState->queueMutex); // Lock the queue for exclusive access
            if (!inferState->textsToStream.empty())
            {

                std::string rawText = inferState->textsToStream.front();
                const std::string textToStream
                    = "data: " + create_return_json(nitro_utils::generate_random_string(20), "_", rawText) + "\n\n";
                inferState->textsToStream.pop();
                lock.unlock(); // Unlock as soon as possible

                // Ensure we do not exceed the buffer size. Truncate if necessary.
                std::size_t bytesToWrite = std::min(nBuffSize, textToStream.size());

                // Copy the text to the provided buffer
                std::memcpy(pBuffer, textToStream.data(), bytesToWrite);
                return bytesToWrite; // Return the number of bytes written to the buffer
            }
            else
            {
                // If the queue is empty, release the lock and wait before trying again
                lock.unlock();
            }
        }
    };

    auto streamResponse = nitro_utils::nitroStreamResponse(chunked_content_provider);
    callback(streamResponse);
    return;
};

// Add definition of your processing function here