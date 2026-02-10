include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(xxHash QUIET)
if (NOT TARGET xxhash)
    FetchContent_Declare(
        xxhash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG v0.8.2
    )
    FetchContent_Populate(xxhash)
    add_subdirectory(${xxhash_SOURCE_DIR}/cmake_unofficial ${xxhash_BINARY_DIR})
endif()

find_package(cpp_tiktoken QUIET)
if (NOT TARGET cpp_tiktoken)
    # We only need cpp_tiktoken for in-tree usage; avoid exporting/installing it.
    set(CPP_TIKTOKEN_INSTALL OFF CACHE BOOL "" FORCE)
    set(CPP_TIKTOKEN_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        cpp_tiktoken
        GIT_REPOSITORY https://github.com/gh-markt/cpp-tiktoken.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(cpp_tiktoken)
endif()

find_package(utf8proc QUIET)
if (NOT TARGET utf8proc)
    # We only need utf8proc for in-tree usage; avoid exporting/installing it.
    set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        utf8proc
        GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
        GIT_TAG v2.9.0
    )
    FetchContent_MakeAvailable(utf8proc)
endif()
