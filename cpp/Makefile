# Makefile for Cortex llamacpp engine - Build, Lint, Test, and Clean
.PHONY: all build package run-e2e-test

RUN_TESTS ?= false
CODE_SIGN ?= false
AZURE_KEY_VAULT_URI ?= xxxx
AZURE_CLIENT_ID ?= xxxx
AZURE_TENANT_ID ?= xxxx
AZURE_CLIENT_SECRET ?= xxxx
AZURE_CERT_NAME ?= xxxx

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex engine
build-deps:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cd tensorrt_llm/cortex.tensorrt-llm; cmake -S ./third-party -B ./build_deps/third-party -DCMAKE_BUILD_TYPE=Release -DCMAKE_OBJECT_PATH_MAX=500; cmake --build ./build_deps/third-party --config Release -j8;"
else
	@cd tensorrt_llm/cortex.tensorrt-llm && cmake -S ./third-party -B ./build_deps/third-party -DCMAKE_BUILD_TYPE=Release -DCMAKE_OBJECT_PATH_MAX=500 && make -C ./build_deps/third-party -j 10 && rm -rf ./build_deps/third-party;
endif

# Build the Cortex engine
build:
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p build; cd build; cmake -GNinja -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded -DCMAKE_POLICY_CMP0141=NEW -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE='Release' -DBUILD_PYT='OFF' -DBUILD_PYBIND='OFF' -DNVTX_DISABLE='ON' -DCMAKE_CUDA_ARCHITECTURES='80-real;86-real;89-real' '-DENABLE_MULTI_DEVICE=0' '-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe' -DBUILD_BENCHMARKS=OFF '-DBUILD_TESTS=OFF' -DTRT_LIB_DIR=C:/workspace/TensorRT-9.3.0.1//lib -DTRT_INCLUDE_DIR=C:/workspace/TensorRT-9.3.0.1//include  ..; cmake --build . --config Release -j2;"
else
	@mkdir -p build && cd build; \
	cmake .. -GNinja -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_BUILD_TYPE='Release' -DBUILD_PYT='OFF' -DBUILD_PYBIND='OFF' -DNVTX_DISABLE='ON' -DCMAKE_CUDA_ARCHITECTURES='80-real;86-real;89-real' '-DENABLE_MULTI_DEVICE=0' '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc' -DBUILD_BENCHMARKS=OFF '-DBUILD_TESTS=OFF' -DTRT_LIB_DIR=/usr/local/tensorrt/lib -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include; \
	cmake --build . --config Release;
endif

pre-package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "echo hello"
else
	echo "hello"
endif

codesign:
ifeq ($(CODE_SIGN),false)
	@echo "Skipping Code Sign"
	@exit 0
endif

ifeq ($(OS),Windows_NT)
	@powershell -Command "dotnet tool install --global AzureSignTool;"
endif

package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "7z a -ttar temp.tar cortex-cpp\*; 7z a -tgzip cortex-cpp.tar.gz temp.tar;"
else
	tar -czvf cortex-cpp.tar.gz cortex-cpp;
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	@powershell -Command "echo hello"
else
	echo "hello"
endif

run-python-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	echo hello
else
	echo hello
endif

clean:
ifeq ($(OS),Windows_NT)
	echo hello
else
	echo "hello"
endif