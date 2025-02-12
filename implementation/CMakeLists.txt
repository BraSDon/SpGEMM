cmake_minimum_required(VERSION 3.22)
enable_testing()

project(SpGEMM VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
# NOTE: This creates the compile_commands.json file, which is needed for the nvim lsp to work with external libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Include Eigen headers
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../external/eigen)

# Main library
add_library(spgemm INTERFACE)
target_sources(spgemm INTERFACE
    src/csr.hpp
    src/spgemm.hpp
)
target_include_directories(spgemm INTERFACE src)
target_compile_features(spgemm INTERFACE cxx_std_20)
target_link_libraries(spgemm INTERFACE OpenMP::OpenMP_CXX)

# Add the matrix generator executable
add_executable(matrix_generator src/matrix_generator.cpp)
target_link_libraries(matrix_generator PRIVATE spgemm)
target_include_directories(matrix_generator PRIVATE src)

if (PROJECT_IS_TOP_LEVEL)
    add_subdirectory(benchmark)
    add_subdirectory(tests)
endif()

