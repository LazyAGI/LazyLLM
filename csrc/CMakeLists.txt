cmake_minimum_required(VERSION 3.16)
project(LazyLLMCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Config lazyllm_bridge lib which defines tmp classes to bridge python and cpp.
file(GLOB_RECURSE LAZYLLM_BRIDGE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/bridge/*.cpp")
add_library(lazyllm_bridge STATIC ${LAZYLLM_BRIDGE_SOURCES})
target_include_directories(lazyllm_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bridge)
target_link_libraries(lazyllm_bridge PUBLIC pybind11::headers Python3::Python)

# Config lazyllm_core lib with pure cpp code.
file(GLOB_RECURSE LAZYLLM_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(lazyllm_core STATIC ${LAZYLLM_CORE_SOURCES})
target_include_directories(lazyllm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lazyllm_core PUBLIC lazyllm_bridge)

# Config lazyllm_cpp lib with binding infomations.
file(GLOB_RECURSE LAZYLLM_BINDING_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/binding/*.cpp")
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
