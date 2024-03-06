#include "tensorrtllm.h"
#include <ostream>
#include <trantor/utils/Logger.h>

void tensorrtllm::chat_completion(
    const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback) const
{
    std::vector<int> text_input = nitro_tokenizer.encode("How to survive in the Abyss chapter 1:\n\n ");
    const int inputLen = text_input.size();
    const std::vector<int> inOutLen = {inputLen, 500}; // input_length, output_length

    const int batchSize = 1;

    std::vector<int32_t> inputIdsHost = text_input;

    std::cout << "Start Nitro testing session: " << std::endl;
    // Input preparation
    auto& bufferManager = gptSession->getBufferManager();
    GenerationInput::TensorPtr inputIds
        = bufferManager.copyFrom(inputIdsHost, ITensor::makeShape({batchSize, inOutLen[0]}), MemoryType::kGPU);

    std::vector<int32_t> inputLengthsHost(batchSize, inOutLen[0]);
    GenerationInput::TensorPtr inputLengths
        = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    bool inputPacked = modelConfig->usePackedInput();

    GenerationInput generationInput{0, 0, inputIds, inputLengths, inputPacked};

    GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
        bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};
    // Define the callback to stream each generated token
    generationOutput.onTokenGenerated = [&bufferManager, inOutLen, this, &generationOutput](
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
                        std::cout<<"\n" <<std::flush;
                }
                else
                {
                        std::cout<< this->nitro_tokenizer.decodeWithSpace(outTok)  <<std::flush;
                }
            }
        }
    };

    gptSession->generate(generationOutput, generationInput, samplingConfig);

    bufferManager.getStream().synchronize();

    LOG_INFO << "Hello world";
    return;
};

// Add definition of your processing function here
