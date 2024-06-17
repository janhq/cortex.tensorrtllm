CMAKE_EXTRA_FLAGS ?= ""
RUN_TESTS ?= true

# Default target, does nothing
all:
	@echo "Specify a target to run"

install-dependencies:
ifeq ($(OS),Windows_NT) # Windows
	cmd /C install_deps.bat
else  # Unix-like systems (Linux and MacOS)
	bash ./install_deps.sh
endif

build-engine:
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p build; cd build; cmake .. $(CMAKE_EXTRA_FLAGS); cmake --build . --config Release;"
else
	# Go to cpp/ dir
	@cd ../../ && \
	mkdir -p build && \
	cd build && \
	cmake .. -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_CORTEX_TENSORRT-LLM=ON -DBUILD_BATCH_MANAGER_DEFAULT=OFF -DCMAKE_CUDA_ARCHITECTURES=89-real -DTRT_LIB_DIR=/usr/local/tensorrt/lib -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include -DCMAKE_BUILD_TYPE=Release && \
	make -j $(nproc)
endif

build-example-server:
ifeq ($(OS),Windows_NT)
else 
	@cd examples/server && \
	mkdir -p build && cd build && \
	cmake .. && cmake --build . --config Release -j12
endif

package:
ifeq ($(OS),Windows_NT)
else
	@mkdir -p cortex.tensorrt-llm && \
	cp ../../build/tensorrt_llm/cortex.tensorrt-llm/libengine.$(shell uname | tr '[:upper:]' '[:lower:]' | sed 's/darwin/dylib/;s/linux/so/') cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer.so.9 cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer.so.9.3.0 cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.9 cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.9.3.0 cortex.tensorrt-llm && \
	cp /home/runner/actions-runner/_work/cortex.tensorrt-llm/cortex.tensorrt-llm/cpp/build/tensorrt_llm/libtensorrt_llm.so cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libnccl.so cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libnccl.so.2 cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libnccl.so.2.19.3 cortex.tensorrt-llm && \
	tar -czvf cortex.tensorrt-llm.tar.gz cortex.tensorrt-llm
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
else
ifeq ($(OS),Windows_NT)
else
	mkdir -p examples/server/build/engines/cortex.tensorrt-llm;
	cp ../../build/tensorrt_llm/cortex.tensorrt-llm/libengine.so examples/server/build/engines/cortex.tensorrt-llm;
	@cd ../../../ && \
	bash ./.github/scripts/e2e-test-server-linux-and-mac.sh "$$(pwd)"
endif
endif
