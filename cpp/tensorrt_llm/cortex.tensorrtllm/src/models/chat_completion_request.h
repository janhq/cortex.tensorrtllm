#pragma once
#include "json/value.h"

namespace tensorrtllm::inferences {
struct ChatCompletionRequest {
  int max_tokens;
  bool stream;
  float top_p;
  float temperature;
  float frequency_penalty;
  float presence_penalty;
  std::string model_id;
  Json::Value messages = Json::Value(Json::arrayValue);
  Json::Value stop = Json::Value(Json::arrayValue);
};

inline ChatCompletionRequest fromJson(std::shared_ptr<Json::Value> json_body) {
  ChatCompletionRequest request;
  if (json_body) {
    request.max_tokens        = json_body->get("max_tokens", 2048).asInt();
    request.stream            = json_body->get("stream", false).asBool();
    request.top_p             = json_body->get("top_p", 0.95).asFloat();
    request.temperature       = json_body->get("temperature", 0.00001f).asFloat();
    request.frequency_penalty = json_body->get("frequency_penalty", 1.3).asFloat();
    request.presence_penalty  = json_body->get("presence_penalty", 0).asFloat();
    request.model_id          = json_body->get("model_id", "default").asString();
    request.messages          = json_body->operator[]("messages");
    request.stop              = json_body->operator[]("stop");
  }
  return request;
}
} // namespace tensorrtllm::inferences
