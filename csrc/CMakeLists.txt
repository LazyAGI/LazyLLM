cmake_minimum_required(VERSION 3.16)
project(LazyLLMCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(LAZYLLM_SOURCE_FILES
    binding/lazyllm.cpp
    binding/doc.cpp
)

set(TARGET_NAME lazyllm_cpp)
pybind11_add_module(${TARGET_NAME} ${LAZYLLM_SOURCE_FILES})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # SHOW_SYMBOL
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET default)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_VISIBILITY_PRESET default)
endif()

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lazyllm)


# TESTS
option(BUILD_TESTS "Build C++ tests" OFF)
if (BUILD_TESTS)
    include(cmake/tests.cmake)
endif ()
