cmake_minimum_required(VERSION 3.5)
project(LazyLLMCPP)

option(SHOW_SYMBOL "Whether to hide symbols" OFF)

set(NEWCXX_ENABLED_FLAG "-std=c++11 -Werror")
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11
                 ${PROJECT_BINARY_DIR}/thirdparty/pybind11)

set(LAZYLLM lazyllm_cpp)

set(LAZYLLM_SOURCE_FILES
    binding/lazyllm.cpp
    binding/doc.cpp
)

pybind11_add_module(${LAZYLLM} SHARED ${LAZYLLM_SOURCE_FILES})

if (SHOW_SYMBOL)
    set_target_properties(${LAZYLLM} PROPERTIES CXX_VISIBILITY_PRESET default)
    set_target_properties(${LAZYLLM} PROPERTIES CUDA_VISIBILITY_PRESET default)
    add_definitions(-DPYBIND11_NAMESPACE=pybind11)
endif()

add_custom_target(liblink ALL
    COMMAND ln -sf ${PROJECT_BINARY_DIR}/$<TARGET_FILE_NAME:${LAZYLLM}> ../../$<TARGET_FILE_NAME:${LAZYLLM}>
)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})

# for test executables
set(LAZYLLM_EXE_FILES
)

foreach (tname ${LAZYLLM_EXE_FILES})
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} ${LAZYLLM} pybind11::module Python::Python)
endforeach (tname)
