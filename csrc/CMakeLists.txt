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

# Create symlink or copy based on platform
if(WIN32)
    # On Windows, use copy instead of symlink (more reliable)
    add_custom_target(liblink ALL
        COMMAND ${CMAKE_COMMAND} -E copy 
        "${PROJECT_BINARY_DIR}/$<TARGET_FILE_NAME:${LAZYLLM}>"
        "${CMAKE_SOURCE_DIR}/../"
    )
else()
    # On Linux/macOS, use symlink
    add_custom_target(liblink ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${PROJECT_BINARY_DIR}/$<TARGET_FILE_NAME:${LAZYLLM}>
            ${CMAKE_SOURCE_DIR}/../$<TARGET_FILE_NAME:${LAZYLLM}>
    )
endif()

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
