cmake_minimum_required(VERSION 3.16)
project(LazyLLMCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Third party libs
include(cmake/third_party.cmake)

# Config lazyllm_core lib with pure cpp code.
file(GLOB_RECURSE LAZYLLM_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp")
add_library(lazyllm_core STATIC ${LAZYLLM_CORE_SOURCES})
target_include_directories(lazyllm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_link_libraries(lazyllm_core PUBLIC xxhash)
target_link_libraries(lazyllm_core PUBLIC tiktoken)
target_link_libraries(lazyllm_core PUBLIC utf8proc)
target_compile_options(lazyllm_core PRIVATE -Werror -Wshadow)

# Config lazyllm_adaptor lib which maintains callback invocations.
file(GLOB_RECURSE LAZYLLM_ADAPTOR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/adaptor/*.cpp")
add_library(lazyllm_adaptor STATIC ${LAZYLLM_ADAPTOR_SOURCES})
target_include_directories(lazyllm_adaptor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/adaptor)
target_link_libraries(lazyllm_adaptor PUBLIC pybind11::headers Python3::Python lazyllm_core)
target_compile_options(lazyllm_adaptor PRIVATE -Werror -Wshadow)

# Config lazyllm_cpp lib with binding infomations.
file(GLOB_RECURSE LAZYLLM_BINDING_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/binding/*.cpp")
set(INTERFACE_TARGET_NAME lazyllm_cpp)
pybind11_add_module(${INTERFACE_TARGET_NAME} ${LAZYLLM_BINDING_SOURCES})
target_link_libraries(${INTERFACE_TARGET_NAME} PRIVATE lazyllm_core lazyllm_adaptor)
target_compile_options(${INTERFACE_TARGET_NAME} PRIVATE -Werror -Wshadow)

# Ensure lazyllm_cpp can find third-party shared libraries under lazyllm/cpp_lib.
set(_lazyllm_cpp_rpath "$ORIGIN/cpp_lib")
if (APPLE)
    set(_lazyllm_cpp_rpath "@loader_path/cpp_lib")
endif()
set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES
    BUILD_RPATH "${_lazyllm_cpp_rpath}"
    INSTALL_RPATH "${_lazyllm_cpp_rpath}"
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # SHOW_SYMBOL
    set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET default)
    set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES CUDA_VISIBILITY_PRESET default)
endif()

# Install
install(TARGETS ${INTERFACE_TARGET_NAME}
    LIBRARY DESTINATION lazyllm COMPONENT lazyllm_cpp
    RUNTIME DESTINATION lazyllm COMPONENT lazyllm_cpp
)
install(TARGETS tiktoken utf8proc
    LIBRARY DESTINATION lazyllm/cpp_lib COMPONENT lazyllm_cpp
    RUNTIME DESTINATION lazyllm/cpp_lib COMPONENT lazyllm_cpp
)


# TESTS
option(BUILD_TESTS "Build C++ tests" OFF)
if (BUILD_TESTS)
    include(cmake/tests.cmake)
endif ()
