import requests
import json
import logging
import time
import concurrent.futures

LLM_MODEL = "/models/Meta-Llama3-8B-Instruct_int4-awq-trt-llm-0.11.0-linux-ampere"
# LLM_MODEL = '/models/OpenHermes-2.5_int4-awq-trt-llm-0.11.0-linux-ampere'
# LLM_MODEL = '/models/Mistral-7B-Instruct-v0.3_int4-awq-trt-llm-0.11.0-linux-ampere'

MAX_TOKENS = 2048
NUM_PARALLEL = 20
BATCH_SIZE = 20
CONST_USER_ROLE = "user"
CONST_ASSISTANT_ROLE = "assistant"

logging.basicConfig(
    filename="./test.log",
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def TestUnloadModel(model):
    new_data = {
        "model": model,
    }

    url_post = "http://127.0.0.1:3928/inferences/tensorrt-llm/unloadmodel"

    res = RequestPost(new_data, url_post)
    if not res:
        exit(1)


def RequestPost(req_data, url, is_stream=False):
    tokens_generated = 0
    try:
        r = requests.post(url, json=req_data, stream=is_stream)
        r.raise_for_status()
        if is_stream:
            if r.encoding is None:
                r.encoding = "utf-8"

            res = ""
            for line in r.iter_lines(decode_unicode=True):
                if line and "[DONE]" not in line:
                    data = json.loads(line[5:])
                    content = data["choices"][0]["delta"]["content"]
                    res += content
                    tokens_generated += 1

            logging.info("{'assistant': '" + res + "'}")
        else:
            res_json = r.json()
            logging.info(res_json)

        if r.status_code == 200:
            return True, tokens_generated
        else:
            logging.warning("{'status_code': " + str(r.status_code) + "}")
            return False, tokens_generated
    except requests.exceptions.HTTPError as error:
        logging.error(error)
        return False, tokens_generated


def TestLoadChatModel():
    new_data = {
        "ctx_len": 2048,
        "model_path": LLM_MODEL,
        "model": "test",
        "n_parallel": NUM_PARALLEL,
        "batch_size": BATCH_SIZE,
    }

    url_post = "http://127.0.0.1:3928/inferences/tensorrt-llm/loadmodel"

    res = RequestPost(new_data, url_post)
    if not res:
        print("Something wrong happened, exit now")
        exit(1)


def TestChatCompletion(prompt):
    content = prompt
    user_msg = [{"role": CONST_USER_ROLE, "content": content}]
    logging.info("{'user': '" + content + "'}")

    new_data = {
        "frequency_penalty": 0,
        "max_tokens": MAX_TOKENS,
        "messages": user_msg,
        "model": LLM_MODEL,
    }

    url_post = "http://127.0.0.1:3928/v1/chat/completions"

    res, count = RequestPost(new_data, url_post, True)
    if not res:
        print("Something wrong happened, exit now")
        exit(1)
    return count


prompts = [
    "What is GPU?",
    "Who won the world cup 2022?",
    "Tell me some dad's joke",
    "Write a quick sort function",
    "What is the price of Nvidia H100?",
    "Who won the world series in 2020?",
    "Tell me a short story",
    "Who is the best football player in the world?",
    "Tell me about compiler",
    "Tell me about AI",
]

if __name__ == "__main__":
    TestLoadChatModel()
    TestChatCompletion("Hello there")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        total_token_processed = 0
        start = time.time()

        futures = [executor.submit(TestChatCompletion, prompt) for prompt in prompts]
        for f in futures:
            total_token_processed += f.result()

        end = time.time()
        print("Model: ", LLM_MODEL)
        print("Requests sent: ", len(futures))
        print("Finished in", end - start, "s")
        print("Total tokens:", total_token_processed)
        print("Throughput:", total_token_processed / (end - start), "tokens/s")
        print(
            "------------------------------------------------------------------------"
        )
