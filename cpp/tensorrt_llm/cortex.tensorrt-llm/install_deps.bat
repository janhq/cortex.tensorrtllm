cmake -S ./third-party -B ./build_deps/third-party
cmake --build ./build_deps/third-party --config Release -- /m:%NUMBER_OF_PROCESSORS%
