cmake_minimum_required(VERSION 3.16)
project(LazyLLMCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Config lazyllm_core lib with pure cpp code.
file(GLOB_RECURSE LAZYLLM_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(lazyllm_core STATIC ${LAZYLLM_CORE_SOURCES})
target_include_directories(lazyllm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Config lazyllm_cpp lib with binding infomations.
set(LAZYLLM_BINDING_SOURCES binding/lazyllm.cpp binding/doc.cpp)
set(INTERFACE_TARGET_NAME lazyllm_cpp)
pybind11_add_module(${INTERFACE_TARGET_NAME} ${LAZYLLM_BINDING_SOURCES})
target_link_libraries(${INTERFACE_TARGET_NAME} PRIVATE lazyllm_core)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # SHOW_SYMBOL
    set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET default)
    set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES CUDA_VISIBILITY_PRESET default)
endif()

# Install
install(TARGETS ${INTERFACE_TARGET_NAME} LIBRARY DESTINATION lazyllm)


# TESTS
option(BUILD_TESTS "Build C++ tests" OFF)
if (BUILD_TESTS)
    include(cmake/tests.cmake)
endif ()
