FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
) # Fix gtest version to maintain C++11 compatibility.

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Resolve libstdc++ from the active C++ compiler. This avoids loading an
# older copy from a preloaded environment (for example Conda) at test runtime.
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
    OUTPUT_VARIABLE LIBSTDCPP_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(TEST_RUNTIME_ENV "")
if (LIBSTDCPP_PATH AND NOT LIBSTDCPP_PATH STREQUAL "libstdc++.so.6")
    # A bare "libstdc++.so.6" means the compiler did not return a concrete path.
    get_filename_component(LIBSTDCPP_DIR "${LIBSTDCPP_PATH}" DIRECTORY)
    if (LIBSTDCPP_DIR)
        # Prepend compiler runtime directory so ctest picks the matching ABI first.
        set(TEST_RUNTIME_ENV "LD_LIBRARY_PATH=${LIBSTDCPP_DIR}:$ENV{LD_LIBRARY_PATH}")
    endif ()
endif ()

file(GLOB_RECURSE LAZYLLM_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

foreach (test_src ${LAZYLLM_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/binding
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${test_name} PRIVATE
        GTest::gtest_main
        lazyllm_core
        pybind11::headers
        Python3::Python
    )
    gtest_add_tests(
        TARGET ${test_name}
        TEST_LIST discovered_tests
    )

    # Attach runtime env per discovered case so each test gets the same loader path.
    if (TEST_RUNTIME_ENV AND discovered_tests)
        set_tests_properties(${discovered_tests} PROPERTIES
            ENVIRONMENT "${TEST_RUNTIME_ENV}"
        )
    endif ()
endforeach ()
