include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://codeload.github.com/google/googletest/zip/refs/tags/release-1.11.0
) # Fix gtest version to maintain C++11 compatibility.

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

file(GLOB_RECURSE LAZYLLM_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

foreach (test_src ${LAZYLLM_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/binding)
    target_link_libraries(${test_name} PRIVATE
        GTest::gtest_main
        pybind11::headers
        Python3::Python
    )
    gtest_discover_tests(${test_name})
endforeach ()
