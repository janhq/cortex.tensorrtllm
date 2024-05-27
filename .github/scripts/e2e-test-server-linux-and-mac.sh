#!/bin/bash

## Example run command
# ./.github/scripts/e2e-test-server-linux-and-mac.sh '/home/jan/cortex.tensorrtllm/'

# Check for required arguments
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path_to_cortex-tensorrtllm>"
    exit 1
fi

CORTEX_TENSORRTLLM_PATH=$1
# Fixed these parameters for testing
DOWNLOAD_MODEL_URL=https://huggingface.co/HHrecode/tllm_checkpoint_1gpu_fp8_hermes_engine/resolve/main/tllm_checkpoint_1gpu_fp8_hermes_engine.tar.gz
MODEL_DIR_PATH=$CORTEX_TENSORRTLLM_PATH/examples/llama/tllm_checkpoint_1gpu_fp8_hermes_engine/
BINARY_DIR_PATH=$CORTEX_TENSORRTLLM_PATH/cpp/tensorrt_llm/nitro/examples/server/build/
BINARY_EXEC_NAME=server

rm /tmp/load-llm-model-res.log /tmp/completion-res.log /tmp/server.log

# Check if model engine exist, if not, download it
if [[ ! -d "$MODEL_DIR_PATH" ]]; then
    pushd $CORTEX_TENSORRTLLM_PATH/examples/llama
    wget $DOWNLOAD_MODEL_URL
    tar -xf tllm_checkpoint_1gpu_fp8_hermes_engine.tar.gz
    popd
fi

# Random port to ensure it's not used
min=10000
max=11000
range=$((max - min + 1))
PORT=$((RANDOM % range + min))

cd $BINARY_DIR_PATH
# Start the binary file
./$BINARY_EXEC_NAME 127.0.0.1 $PORT >/tmp/server.log &

# Get the process id of the binary file
pid=$!

if ! ps -p $pid >/dev/null; then
    echo "server failed to start. Logs:"
    cat /tmp/server.log
    exit 1
fi

# Wait for second to let the server start
sleep 2

# Run the curl commands
response1=$(curl --connect-timeout 60 -o /tmp/load-llm-model-res.log -s -w "%{http_code}" --location "http://127.0.0.1:$PORT/inferences/tensorrtllm/loadmodel" \
    --header 'Content-Type: application/json' \
    --data '{
        "engine_path": "'$MODEL_DIR_PATH'",
        "ctx_len": 512,
        "ngl": 100
    }'
)

response2=$(curl --connect-timeout 60 -o /tmp/completion-res.log -s -w "%{http_code}" --location "http://127.0.0.1:$PORT/v1/chat/completions" \
    --header 'Content-Type: application/json' \
    --header 'Accept: text/event-stream' \
    --header 'Access-Control-Allow-Origin: *' \
    --data '{
        "messages": [{
            "content": "Hello there 👋",
            "role": "assistant"
        },
        {
            "content": "Write a long story about NVIDIA!!!!",
            "role": "user"
        }],
        "stream": true,
        "max_tokens": 512
    }'
)

error_occurred=0
if [[ "$response1" -ne 200 ]]; then
    echo "The load llm model curl command failed with status code: $response1"
    cat /tmp/load-llm-model-res.log
    error_occurred=1
fi

if [[ "$response2" -ne 200 ]]; then
    echo "The completion curl command failed with status code: $response2"
    cat /tmp/completion-res.log
    error_occurred=1
fi

if [[ "$error_occurred" -eq 1 ]]; then
    echo "Server test run failed!!!!!!!!!!!!!!!!!!!!!!"
    echo "Server Error Logs:"
    cat /tmp/server.log
    kill $pid
    exit 1
fi

echo "----------------------"
echo "Log load model:"
cat /tmp/load-llm-model-res.log

echo "----------------------"
echo "Log run test:"
cat /tmp/completion-res.log

# Kill the server process
kill $pid
# Wait for a few seconds for the server to complete logging
sleep 3

echo "----------------------"
echo "Log server:"
cat /tmp/server.log

echo "Server test run successfully!"
