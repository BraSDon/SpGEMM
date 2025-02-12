# Gustavson SpGEMM
This repository contains two implementations of the Gustavson Sparse Matrix Multiplication algorithm - one sequential and the other parallel via OpenMP.

## Getting started
### Pre-requisites
1. **CMake:** Version 3.22 or higher
2. **C++ Compiler:** C++20 or higher
3. (Optional) Maybe some other system libraries to run tsan, msan and ubsan
4. (Optional) Datasets for benchmarking: [Florida Sparse Matrices](http://sparse.tamu.edu/)
   - nemeth07
   - lhr71c
   - c-71
   - preferentialAttachment
   - consph
   - rgg_2_22_s0
   - rajat31
   - M6
   - ASIC_680ks

### Running
1. Clone the repository
    ```sh
    git clone https://github.com/BraSDon/SpGEMM.git
    ```
2. Change into the project directory
    ```sh
    cd SpGEMM/
    ```
3. Initialize submodules
    ```sh
    git submodule update --init --recursive
    ```
4. Initialize the build directory
    ```sh
    cmake --preset default
    ```
5. Build the project
   ```sh
    cmake --build --preset default
    ```
6. Optional: run the tests
   ```sh
    cd build/
    ctest
    ```
7. Optional: run the benchmarks (ensure you have the downloaded the datasets into implementation/data)
   ```sh
    cd build/benchmark/
    ./bench
    ```
   or if you want plots
    ```sh
    cmake --build --preset release -t plots
    ```
