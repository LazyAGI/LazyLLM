include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
) # Fix gtest version to maintain C++17 compatibility.

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
    # Ensure tests use the same libstdc++ as the compiler, avoiding conda-incompatible versions.
    if (LAZYLLM_LIBSTDCPP_DIR)
        set_target_properties(${test_name} PROPERTIES
            BUILD_RPATH "${LAZYLLM_LIBSTDCPP_DIR}"
        )
        if (NOT WIN32 AND NOT APPLE)
            target_link_options(${test_name} PRIVATE -Wl,--disable-new-dtags)
        endif ()
    endif ()
    gtest_discover_tests(
        ${test_name}
        TEST_LIST discovered_tests
    )
endforeach ()
