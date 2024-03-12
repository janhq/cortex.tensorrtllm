# Build engine for NVIDIA TensorRT-LLM on Windows

## Automated script

- Open Powershell (Administrator)
- Run command:

```ps1
.\TensorRT-LLM\windows\setup_env.ps1
```

- CUDA 12.2
- Python 3.10
- Miscrosoft MPI
- TensorRT 9.2
- CuDNN 8.9

## Manual step by step

- Step 1: Prepare environment with command

```ps1
.\TensorRT-LLM\windows\setup_env.ps1
```

- Step 2: Verify that TensorRT-LLM and TensorRT can run fine on your machine by running these 2 command

```ps1
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
```

- Step 2: Download huggingface model

```ps1
pip install -U "huggingface_hub[cli]" hf_transfer
# Create folder to store model
mkdir model
mkdir checkpoint
mkdir engine

# Download model to the folder `model\` with huggingface model handler (e.g jan-hq/stealth-v1.2)
set HF_HUB_ENABLE_HF_TRANSFER=1 && huggingface-cli download --repo-type model --local-dir .\model <model handler>

cd examples\llama
# At this step, you can choose whether to run the model at INT4 for FP16, choose either next step

# For FP16 option
python convert_checkpoint.py --model_dir ..\..\..\model --output_dir ..\..\..\checkpoint --dtype float16

# For INT4 option (currently it has a problem with `nvidia-anmo`, have to check)
python ../quantization/quantize.py --model_dir ..\..\..\model --output_dir ..\..\..\checkpoint --dtype float16 --dtype float16 --awq_block_size 128 --kv_cache_dtype int8 --calib_size 32

# Build the engine for TensorRT-LLM to use from either FP16 or INT4 options
trtllm-build --checkpoint_dir ..\..\..\checkpoint --output_dir ..\..\..\engine --gemm_plugin float16

# Prepare the tokenizers related files
cp ..\..\..\model/tokenizer.json ..\..\..\engine
cp ..\..\..\model/tokenizer.model ..\..\..\engine
cp ..\..\..\model/tokenizer_config.json ..\..\..\engine
```

- Step 3: The `engine` folder is ready to be used with TensorRT-LLM engine
