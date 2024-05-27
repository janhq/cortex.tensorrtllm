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
	cmake .. -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_NITRO=ON -DBUILD_BATCH_MANAGER_DEFAULT=OFF -DCMAKE_CUDA_ARCHITECTURES=89-real -DTRT_LIB_DIR=/usr/local/tensorrt/lib -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include -DCMAKE_BUILD_TYPE=Release && \
	make -j $(nproc)
endif

build-example-server:
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p .\examples\server\build\engines\cortex.python; cd .\examples\server\build; cmake .. $(CMAKE_EXTRA_FLAGS); cmake --build . --config Release; cp -r ..\..\..\build\python .\Release\engines\cortex.python\python"
else 
	@cd examples/server && \
	mkdir -p build && cd build && \
	cmake .. && cmake --build . --config Release -j12
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
else
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p .\examples\server\build\Release\engines\cortex.python; cd examples\server\build\Release; cp ..\..\..\..\build\Release\engine.dll engines\cortex.python; ..\..\..\..\.github\scripts\e2e-test-server-windows.bat server.exe ..\..\..\..\$(PYTHON_FILE_EXECUTION_PATH);"
else
	mkdir -p examples/server/build/engines/cortex.tensorrtllm;
	cp ../../build/tensorrt_llm/nitro/libengine.so examples/server/build/engines/cortex.tensorrtllm;
	@cd ../../../ && \
	bash ./.github/scripts/e2e-test-server-linux-and-mac.sh "$$(pwd)"
endif
endif
