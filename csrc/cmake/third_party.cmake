include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
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

find_package(sentencepiece QUIET)
if (NOT TARGET sentencepiece AND NOT TARGET sentencepiece::sentencepiece AND NOT TARGET sentencepiece-static)
    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.2.0
    )
    FetchContent_MakeAvailable(sentencepiece)
endif()
if (TARGET sentencepiece::sentencepiece)
    add_library(sentencepiece ALIAS sentencepiece::sentencepiece)
elseif (TARGET sentencepiece)
    add_library(sentencepiece ALIAS sentencepiece)
elseif (TARGET sentencepiece-static)
    add_library(sentencepiece ALIAS sentencepiece-static)
endif()
