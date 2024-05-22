# Makefile for Cortex python-runtime engine - Build, Lint, Test, and Clean

CMAKE_EXTRA_FLAGS ?= ""
RUN_TESTS ?= true
PYTHON_FILE_EXECUTION_PATH ?= .github/scripts/python-file-to-test.py

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex python-runtime engine
install-dependencies:
ifeq ($(OS),Windows_NT) # Windows
	cmd /C install_deps.bat
else  # Unix-like systems (Linux and MacOS)
	bash ./install_deps.sh
endif
