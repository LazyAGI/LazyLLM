# Download tokenizer model files from authoritative sources at configure time.
set(TOKENIZERS_DIR "${CMAKE_BINARY_DIR}/tokenizers")
file(MAKE_DIRECTORY ${TOKENIZERS_DIR})

set(OPENAI_TOKENIZER_FILES
    r50k_base.tiktoken
    p50k_base.tiktoken
    cl100k_base.tiktoken
    o200k_base.tiktoken
)

foreach(file ${OPENAI_TOKENIZER_FILES})
    set(url "https://openaipublic.blob.core.windows.net/encodings/${file}")
    set(dest "${TOKENIZERS_DIR}/${file}")
    if(NOT EXISTS ${dest})
        message(STATUS "Downloading tokenizer ${file}")
        file(DOWNLOAD ${url} ${dest} STATUS download_status)
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            list(GET download_status 1 status_msg)
            message(FATAL_ERROR "Failed to download ${file} from ${url}: ${status_msg}")
        endif()
    endif()
endforeach()

set(CPP_TIKTOKEN_COMMIT "9323db528d52e48900c75ce197c3251085b18480")
set(CPP_TIKTOKEN_FILES
    qwen.tiktoken
    tokenizer.model
    tokenizer_llama3.1.model
)

foreach(file ${CPP_TIKTOKEN_FILES})
    set(url "https://raw.githubusercontent.com/gh-markt/cpp-tiktoken/${CPP_TIKTOKEN_COMMIT}/${file}")
    set(dest "${TOKENIZERS_DIR}/${file}")
    if(NOT EXISTS ${dest})
        message(STATUS "Downloading tokenizer ${file}")
        file(DOWNLOAD ${url} ${dest} STATUS download_status)
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            list(GET download_status 1 status_msg)
            message(FATAL_ERROR "Failed to download ${file} from ${url}: ${status_msg}")
        endif()
    endif()
endforeach()
