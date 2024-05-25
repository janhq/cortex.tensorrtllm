#include "src/models/chat_completion_request.h"
#include "src/models/load_model_request.h"
#include "src/utils/tensorrtllm_utils.h"
#include <climits> // for PATH_MAX
#include <cstddef>
#include <iostream>
#include <json/value.h>
#include <trantor/utils/Logger.h>

#include "src/tensorrtllm_engine.h"

#if defined(__APPLE__) && defined(__MACH__)
#include <libgen.h> // for dirname()
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <libgen.h> // for dirname()
#include <unistd.h> // for readlink()
#elif defined(_WIN32)
#include <windows.h>
#undef max
#else
#error "Unsupported platform!"
#endif

using namespace tensorrtllm;

int main(int argc, char* argv[])
{
    TensorrtllmEngine engine;

    model::LoadModelRequest mock_load_model;
    mock_load_model.engine_path = "/root/nitro-tensorrt-llm/examples/llama/tllm_checkpoint_1gpu_fp8_hermes_engine";
    mock_load_model.ctx_len = 512;
    engine.LoadModelImpl(std::move(mock_load_model), nullptr);
    LOG_DEBUG << "loadModel done!!!";

    Json::Value asistant_message;
    asistant_message["content"] = "Hello there";
    asistant_message["role"] = "assistant";
    Json::Value user_message;
    user_message["content"] = "Write a long story about NVIDIA!!!!";
    user_message["role"] = "user";
    inferences::ChatCompletionRequest mock_chat_completion;
    // mock_chat_completion.messages.append(asistant_message);
    mock_chat_completion.messages.append(user_message);
    mock_chat_completion.stream = true;
    mock_chat_completion.max_tokens = 2048;
    engine.HandleChatCompletionImpl(std::move(mock_chat_completion), nullptr);
    LOG_DEBUG << "chat_completion done!!!";

    return 0;
}
