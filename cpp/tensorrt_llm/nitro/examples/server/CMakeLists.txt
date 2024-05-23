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
# NItro init
cmake_minimum_required(VERSION 3.5)
project(server)
find_package(Threads REQUIRED)

# if(UNIX AND NOT APPLE)
#   set(LINKER_FLAGS -ldl)
# endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

include(CheckIncludeFileCXX)
# CPP version
check_include_file_cxx(any HAS_ANY)
check_include_file_cxx(string_view HAS_STRING_VIEW)
check_include_file_cxx(coroutine HAS_COROUTINE)
if(HAS_ANY
  AND HAS_STRING_VIEW
  AND HAS_COROUTINE)
  set(CMAKE_CXX_STANDARD 20)
elseif(HAS_ANY AND HAS_STRING_VIEW)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPENSSL_USE_STATIC_LIBS TRUE)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build_deps/_install)
set(CORTEX_COMMON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../base/)

find_library(JSONCPP
    NAMES jsoncpp
    HINTS "${THIRD_PARTY_PATH}/lib"
)

find_package(PkgConfig REQUIRED)
find_package(Drogon CONFIG REQUIRED)

if(NOT WIN32) # Linux
  # Use pkg-config to find the SentencePiece library
  pkg_search_module(SENTENCEPIECE REQUIRED sentencepiece)
else() # Windows
  set(SENTENCEPIECE_INCLUDE_DIRS "${CMAKE_PREFIX_PATH}/include")
  set(SENTENCEPIECE_LIBRARY_DIRS "${CMAKE_PREFIX_PATH}/lib")
endif()

message(STATUS "SentencePiece library dirs: ${SENTENCEPIECE_LIBRARY_DIRS}")
message(STATUS "SentencePiece header dirs: ${SENTENCEPIECE_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include ${SENTENCEPIECE_INCLUDE_DIRS})

link_directories(${SENTENCEPIECE_LIBRARY_DIRS})

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(server_proj)

add_executable(server
    server.cc
    dylib.h
    httplib.h
)

target_link_libraries(server PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm cxxopts::cxxopts sentencepiece 
                             PRIVATE ${JSONCPP} Drogon::Drogon ${LINKER_FLAGS} ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(server PRIVATE cxx_std_17)
target_compile_definitions(server PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

target_include_directories(server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                          ${CORTEX_COMMON_PATH}
                                          ${THIRD_PARTY_PATH}/include)

add_dependencies(server_proj server)
