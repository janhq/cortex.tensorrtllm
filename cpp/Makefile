# Makefile for Cortex llamacpp engine - Build, Lint, Test, and Clean
.PHONY: all build package run-e2e-test


CMAKE_EXTRA_FLAGS ?= ""
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
	@powershell -Command "& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64; cd tensorrt_llm/nitro; cmake -S ./nitro_deps -B ./build_deps/nitro_deps; cmake --build ./build_deps/nitro_deps --config Release -j4;"
else
	@./install_deps.sh;
	@mkdir -p build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j4;
endif

# Build the Cortex engine
build:
ifeq ($(OS),Windows_NT)
	@powershell -Command "& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64; mkdir -p build; cd build; cmake -GNinja -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded -DCMAKE_POLICY_CMP0141=NEW -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_BUILD_TYPE='Release' -DBUILD_PYT='OFF' -DBUILD_PYBIND='OFF' -DNVTX_DISABLE='ON' -DCMAKE_CUDA_ARCHITECTURES='86-real;89-real;90-real' '-DENABLE_MULTI_DEVICE=0' '-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe' '-DBUILD_NITRO=ON' -DBUILD_BENCHMARKS=OFF '-DBUILD_TESTS=OFF' -DTRT_LIB_DIR=C:/workspace/TensorRT-10.0.1.6//lib -DTRT_INCLUDE_DIR=C:/workspace/TensorRT-10.0.1.6//include  ..; cmake --build . --parallel 2 --config Release;"
else
	@./install_deps.sh;
	@mkdir -p build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j4;
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