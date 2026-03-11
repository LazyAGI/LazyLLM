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

# Runtime loader configuration per platform.
set(_lazyllm_cpp_rpath "")
set(LAZYLLM_TEST_RUNTIME_ENV "" CACHE INTERNAL "Runtime env for LazyLLM C++ tests" FORCE)
if (WIN32)
    # Windows has no ELF rpath; loader resolution is driven by PATH and DLL search order.
    # Keep test runtime env empty by default.
elseif (APPLE)
    # Ensure lazyllm_cpp can find third-party dylibs under lazyllm/cpp_lib.
    list(APPEND _lazyllm_cpp_rpath "@loader_path/cpp_lib")
else ()
    # Ensure lazyllm_cpp can find third-party shared libraries under lazyllm/cpp_lib.
    list(APPEND _lazyllm_cpp_rpath "$ORIGIN/cpp_lib")
    # Use DT_RPATH (instead of DT_RUNPATH) so the extension's own runtime
    # search path can take precedence over host interpreter bundled libs.
    target_link_options(${INTERFACE_TARGET_NAME} PRIVATE -Wl,--disable-new-dtags)

    # Resolve libstdc++ from the active C++ compiler and include it in rpath.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
        OUTPUT_VARIABLE LIBSTDCPP_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (LIBSTDCPP_PATH AND NOT LIBSTDCPP_PATH STREQUAL "libstdc++.so.6")
        get_filename_component(LIBSTDCPP_DIR "${LIBSTDCPP_PATH}" DIRECTORY)
        if (LIBSTDCPP_DIR)
            list(APPEND _lazyllm_cpp_rpath "${LIBSTDCPP_DIR}")
            set(LAZYLLM_TEST_RUNTIME_ENV "LD_LIBRARY_PATH=${LIBSTDCPP_DIR}:$ENV{LD_LIBRARY_PATH}"
                CACHE INTERNAL "Runtime env for LazyLLM C++ tests" FORCE)
        endif ()
    endif ()
endif ()

if (_lazyllm_cpp_rpath)
    set_target_properties(${INTERFACE_TARGET_NAME} PROPERTIES
        BUILD_RPATH "${_lazyllm_cpp_rpath}"
        INSTALL_RPATH "${_lazyllm_cpp_rpath}"
    )
endif ()

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
