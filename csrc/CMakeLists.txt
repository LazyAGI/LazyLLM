cmake_minimum_required(VERSION 3.16)
project(LazyLLMCPP LANGUAGES CXX)

option(SHOW_SYMBOL "Whether to hide symbols" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(LAZYLLM lazyllm_cpp)

set(LAZYLLM_SOURCE_FILES
    binding/lazyllm.cpp
    binding/doc.cpp
)

pybind11_add_module(${LAZYLLM} ${LAZYLLM_SOURCE_FILES})

if (SHOW_SYMBOL)
    set_target_properties(${LAZYLLM} PROPERTIES CXX_VISIBILITY_PRESET default)
    set_target_properties(${LAZYLLM} PROPERTIES CUDA_VISIBILITY_PRESET default)
    add_definitions(-DPYBIND11_NAMESPACE=pybind11)
endif()

# for test executables
# set(LAZYLLM_EXE_FILES)

# foreach (tname ${LAZYLLM_EXE_FILES})
#     add_executable(${tname} ${tname}.cpp)
#     target_link_libraries(${tname} ${LAZYLLM} pybind11::module Python::Python)
# endforeach (tname)

install(TARGETS ${LAZYLLM}
        LIBRARY DESTINATION lazyllm)
