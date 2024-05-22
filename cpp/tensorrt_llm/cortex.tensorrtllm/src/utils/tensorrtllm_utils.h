#pragma once

#include "json/writer.h"
#include <filesystem>
#include <functional>
#include <random>

#include "models/chat_completion_request.h"
#include "models/load_model_request.h"


namespace tensorrtllm_utils {

inline bool ValidateLoadModelRequest(
    const tensorrtllm::model::LoadModelRequest& request,
    const std::function<void(Json::Value&&, Json::Value&&)>& callback) {

  return true;
}

inline bool ValidateLoadModelFiles(
    std::string engine_dir_path, 
    std::filesystem::path& engine_dir,
    std::filesystem::path& config_file,
    std::filesystem::path& tokenizer_file) {

  engine_dir = std::filesystem::path(engine_dir_path);
  config_file = engine_dir / "config.json";
  tokenizer_file = engine_dir / "tokenizer.model";
  return true;
}

inline bool ValidateHandleChatCompletionRequest(
    const tensorrtllm::inferences::ChatCompletionRequest& request,
    const std::function<void(Json::Value&&, Json::Value&&)>& callback) {

  return true;
}

inline std::string GenerateRandomString(std::size_t length) {
    const std::string characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distribution(0, characters.size() - 1);
    std::string random_string(length, '\0');
    std::generate_n(random_string.begin(), length, [&]() { return characters[distribution(generator)]; });
    return random_string;
}

// Only support single token stopping point now
inline std::string CreateReturnJson(std::string const& id, std::string const& model, std::string const& content, 
                                    Json::Value finish_reason = Json::Value()) {
    Json::Value root;
    root["id"] = id;
    root["model"] = model;
    root["created"] = static_cast<int>(std::time(nullptr));
    root["object"] = "chat.completion.chunk";

    Json::Value choices_array(Json::arrayValue);
    Json::Value choice;

    choice["index"] = 0;
    Json::Value delta;
    delta["content"] = content;
    choice["delta"] = delta;
    choice["finish_reason"] = finish_reason;

    choices_array.append(choice);
    root["choices"] = choices_array;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = ""; // This sets the indentation to an empty string,
                                // producing compact output.
    return Json::writeString(writer, root);
}

} // namespace tensorrtllm
