cmake_minimum_required(VERSION 3.1.2)

project(tiktoken LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build_deps/_install)

find_library(PCRE2
    NAMES pcre2-8
    HINTS "${CMAKE_PREFIX_PATH}/lib"
)

option(CPP_TIKTOKEN_TESTING "Enable testing" OFF)

set(OPENAPI_SOURCES byte_pair_encoding.cc emdedded_resource_reader.cc modelparams.cc encoding.cc encoding_utils.cc pcre2_regex.cc)

add_library(tiktoken ${OPENAPI_SOURCES})
include_directories(${CMAKE_PREFIX_PATH}/include)
link_directories(${CMAKE_PREFIX_PATH}/lib)

target_link_libraries(tiktoken pcre2-8)

target_include_directories(tiktoken PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_PREFIX_PATH}/include)

if (NOT CPP_TIKTOKEN_TESTING)
    message(STATUS "Tests off")
else()
    add_subdirectory(ut)
endif()

MESSAGE(STATUS "Copying tokenizers to '${CMAKE_BINARY_DIR}/tokenizers'.")
FILE(COPY o200k_base.tiktoken cl100k_base.tiktoken p50k_base.tiktoken r50k_base.tiktoken tokenizer.model DESTINATION "${CMAKE_BINARY_DIR}/tokenizers")
MESSAGE(STATUS "Tokenizers copied.")