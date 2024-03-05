# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_deps/_install)

message(STATUS "Current Source Directory NITRO: ${CMAKE_CURRENT_SOURCE_DIR}")

# Enable pkg-config support in CMake
find_package(PkgConfig REQUIRED)

# Use pkg-config to find the SentencePiece library
pkg_search_module(SENTENCEPIECE REQUIRED sentencepiece)


include_directories(${PROJECT_SOURCE_DIR}/include ${SENTENCEPIECE_INCLUDE_DIRS})

link_directories(${SENTENCEPIECE_LIBRARY_DIRS})

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(nitro_proj)

set(CXXOPTS_SRC_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cxxopts)
add_subdirectory(${CXXOPTS_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)

add_executable(nitro main.cc)

target_link_libraries(
  nitro PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm cxxopts::cxxopts ${SENTENCEPIECE_LIBRARIES})

target_compile_features(nitro PRIVATE cxx_std_17)
target_compile_definitions(nitro PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

add_dependencies(nitro_proj nitro)

