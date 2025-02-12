cmake_minimum_required(VERSION 3.22)

# Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "")
add_subdirectory(../../external/googlebenchmark googlebenchmark EXCLUDE_FROM_ALL)

# Add Eigen include directory
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../external/eigen)

add_executable(bench benchmark.cpp)
target_compile_features(bench PRIVATE cxx_std_20)
target_compile_definitions(bench PRIVATE LIB_USAGE)
target_compile_options(bench PRIVATE -O3 -g -march=native -DNDEBUG)
target_link_libraries(bench PRIVATE spgemm benchmark::benchmark_main)

# Plots
add_custom_command(OUTPUT benchmark.json
                COMMAND bench
                ARGS --benchmark_out=benchmark.json --benchmark_out_format=json
                DEPENDS bench
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                COMMENT "Running benchmark"
                VERBATIM)


add_custom_command(OUTPUT plots.pdf
                COMMAND Rscript "${PROJECT_SOURCE_DIR}/eval/make_plots.R"
                DEPENDS "${PROJECT_SOURCE_DIR}/eval/make_plots.R" benchmark.json
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                COMMENT "Generating plots"
                VERBATIM)
add_custom_target(plots DEPENDS plots.pdf)
