cmake_minimum_required(VERSION 3.22)

# Tests
include(CTest)
if(BUILD_TESTING)
    set(BUILD_GMOCK OFF CACHE BOOL "")
    set(INSTALL_GTEST OFF CACHE BOOL "")
    add_subdirectory(../../external/googletest googletest EXCLUDE_FROM_ALL)
    include(GoogleTest)

    # Add Eigen include directory
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../external/eigen)

    # CSR tests
    add_executable(csr csr_tests.cpp)
    target_compile_features(csr PRIVATE cxx_std_20)
    target_link_libraries(csr PRIVATE spgemm GTest::gtest_main)
    gtest_discover_tests(csr)

    # SpGEMM tests
    add_executable(spgemm_test spgemm_tests.cpp)
    target_compile_features(spgemm_test PRIVATE cxx_std_20)
    target_compile_definitions(spgemm_test PRIVATE LIB_USAGE)
    target_link_libraries(spgemm_test PRIVATE spgemm GTest::gtest_main)
    gtest_discover_tests(spgemm_test)

    # Matrix generator tests
    add_executable(matrix_generator_test matrix_generator_tests.cpp)
    target_compile_features(matrix_generator_test PRIVATE cxx_std_20)
    target_compile_definitions(matrix_generator_test PRIVATE LIB_USAGE)
    target_link_libraries(matrix_generator_test PRIVATE spgemm GTest::gtest_main)
    gtest_discover_tests(matrix_generator_test)
endif()
