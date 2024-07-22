cmake_minimum_required(VERSION 3.22)  # Required for FetchContent

project(MyProject)

include(ExternalProject)

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

# Define variables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THIRD_PARTY_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../build_deps/_install)
#if(NOT THIRD_PARTY_INSTALL_PATH )
#  message(FATAL_ERROR "TRITON_THIRD_PARTY_INSTALL_PREFIX must be set")
#endif() # TRITON_THIRD_PARTY_INSTALL_PREFIX
# To force the find_package to look for .a inside self installed version
#set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#
# Add the external project
set(ZLIB_USE_STATIC_LIBS OFF)
find_package(ZLIB)
if(NOT ZLIB_FOUND)
    set(ZLIB_USE_STATIC_LIBS ON)
    ExternalProject_Add(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.11
        CMAKE_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
    )
endif()

ExternalProject_Add(
    brotli
    GIT_REPOSITORY https://github.com/google/brotli
    GIT_TAG v1.1.0
    CMAKE_ARGS 
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DSHARE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/share
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
    GIT_TAG 1.9.5
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    c-ares
    GIT_REPOSITORY https://github.com/c-ares/c-ares
    GIT_TAG cares-1_26_0
    CMAKE_ARGS
    	-DCARES_SHARED=OFF
	    -DCARES_STATIC=ON
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    trantor
    GIT_REPOSITORY https://github.com/an-tao/trantor
    GIT_TAG v1.5.17
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PATH}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG v0.2.0
    CMAKE_ARGS
    	-DSPM_ENABLE_SHARED=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 11.0.1
    CMAKE_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

if(WIN32)
  SET(PCRE2_SHARED_LIBS ON)
  SET(PCRE2_STATIC_LIBS OFF)
else()
  SET(PCRE2_SHARED_LIBS OFF)
  SET(PCRE2_STATIC_LIBS ON)
endif()

ExternalProject_Add(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2
    GIT_TAG pcre2-10.44
    CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=${PCRE2_SHARED_LIBS}
            -DBUILD_STATIC_LIBS=${PCRE2_STATIC_LIBS}
            # -DPCRE2_STATIC_RUNTIME=ON
            -DPCRE2_STATIC_PIC=ON
            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

# if(UNIX AND NOT APPLE)
# set(PCRE2_SRC_PATH ${THIRD_PARTY_INSTALL_PATH}/../third-party/pcre2-prefix/src/pcre2)

# # The configure command does not work
# # Patch the cmake file to allow CFLAGS
# ExternalProject_Add_Step(
#     pcre2 patch_cmake
#     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/pcre2-patch/CMakeLists.txt ${PCRE2_SRC_PATH}/CMakeLists.txt
#     DEPENDEES build
#   )
# endif()

include_directories(${THIRD_PARTY_INSTALL_PATH}/include)
link_directories(${THIRD_PARTY_INSTALL_PATH}/lib)
# Optionally link or add dependencies to your targets
add_dependencies(trantor c-ares jsoncpp brotli fmt pcre2)

# target_link_libraries(<your-target> ...)
