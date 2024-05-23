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
project(cortex.tensorrtllm)
set(TARGET engine)

add_library(${TARGET} SHARED src/tensorrtllm_engine.cc)

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

include(CheckIncludeFileCXX)
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
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(OPENSSL_USE_STATIC_LIBS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../nitro/build_deps/_install)

message(STATUS "Current Cmake third party path: ${THIRD_PARTY_PATH}")

find_library(JSONCPP
    NAMES jsoncpp
    HINTS "${THIRD_PARTY_PATH}/lib"
)

find_package(Drogon CONFIG REQUIRED)

if(NOT WIN32) # Linux
    # # Use pkg-config to find the SentencePiece library
    # find_package(PkgConfig REQUIRED)
    # # pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
    # pkg_search_module(SENTENCEPIECE REQUIRED sentencepiece)
    find_library(SENTENCEPIECE
        NAMES sentencepiece
        HINTS "${THIRD_PARTY_PATH}/lib"
    )
else() # Windows
    set(SENTENCEPIECE_INCLUDE_DIRS "${THIRD_PARTY_PATH}/include")
    set(SENTENCEPIECE_LIBRARY_DIRS "${THIRD_PARTY_PATH}/lib")
endif()

message(STATUS "SentencePiece library dirs: ${SENTENCEPIECE_LIBRARY_DIRS}")
message(STATUS "SentencePiece header dirs: ${SENTENCEPIECE_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include ${SENTENCEPIECE_INCLUDE_DIRS})

link_directories(${SENTENCEPIECE_LIBRARY_DIRS})

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/../../..")

# add_custom_target(nitro_proj)
# add_custom_target(cortex-tensorllm_proj)

# !!!WARNING!!! Un-comment these CXXOPTS related lines after removing nitro
# set(CXXOPTS_SRC_DIR ${TOP_LEVEL_DIR}/3rdparty/cxxopts)
# add_subdirectory(${CXXOPTS_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)

set(PLUGINS_SRC_DIR ${TOP_LEVEL_DIR}/cpp/build/tensorrt_llm/plugins)
link_directories(${PLUGINS_SRC_DIR})

set(TENSORRT_LLM_LIB ${TOP_LEVEL_DIR}/cpp/include/)

target_link_libraries(
    ${TARGET} PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm cxxopts::cxxopts ${SENTENCEPIECE} PRIVATE ${JSONCPP} Drogon::Drogon ${CMAKE_THREAD_LIBS_INIT} )

target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

# aux_source_directory(controllers CTL_SRC)
# aux_source_directory(common COMMON_SRC)
# aux_source_directory(context CONTEXT_SRC)
# aux_source_directory(models MODEL_SRC)
# target_sources(nitro PRIVATE ${CTL_SRC} ${COMMON_SRC} ${CONTEXT_SRC})

target_include_directories(${TARGET} PRIVATE 
                           ${CMAKE_CURRENT_SOURCE_DIR} 
                           ${CMAKE_CURRENT_SOURCE_DIR}/src/ 
                           ${THIRD_PARTY_PATH}/include/
                           ${TENSORRT_LLM_LIB})
# ${CMAKE_CURRENT_SOURCE_DIR}/models)


# add_dependencies(nitro_proj nitro)
# add_dependencies(cortex-tensorllm_proj cortex-tensorllm)

