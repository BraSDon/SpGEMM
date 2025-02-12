#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

#include "common.h"
#include "csr.hpp"

template <class T>
class DistributionHelper {
public:
    DistributionHelper(T min = -5, T max = 5) {
        if constexpr (T_is_int) {
            uniform_int_dist_ = std::uniform_int_distribution<int>(static_cast<int>(min), static_cast<int>(max));
        } else if constexpr (T_is_double) {
            uniform_double_dist_ = std::uniform_real_distribution<double>(min, max);
        } else {
            static_assert(std::is_same<T, void>::value, "Unsupported type for DistributionHelper.");
            throw std::invalid_argument("Unsupported type.");
        }
    }

    T generate(std::mt19937_64& rng) {
        if constexpr (T_is_int) {
            return uniform_int_dist_(rng);
        } else if constexpr (T_is_double) {
            return uniform_double_dist_(rng);
        } else {
            throw std::invalid_argument("Unsupported type.");
        }
    }

private:
    std::uniform_int_distribution<int> uniform_int_dist_;
    std::uniform_real_distribution<double> uniform_double_dist_;

    static constexpr bool T_is_int = std::is_same<T, int>::value;
    static constexpr bool T_is_double = std::is_same<T, double>::value;
};

// NOTE: only supports int and double
template <class T>
class MatrixGenerator {
private:
    std::mt19937_64 rng_;
    DistributionHelper<T> dist_;
public:
    MatrixGenerator(unsigned long seed, T min = -5, T max = 5) : rng_(seed), dist_(min, max) {}
    void reset_seed(unsigned long seed) {
        rng_.seed(seed);
    }

    // NOTE: triples are not sorted
    TripleMatrix<T> generate(size_t rows, size_t cols, double sparsity) {
        TripleMatrix<T> triples;
        auto nnz = static_cast<size_t>(static_cast<double>(rows * cols) * (1 - sparsity));
        triples.reserve(nnz);
        for (size_t i = 0; i < nnz; ++i) {
            size_t row = rng_() % rows;
            size_t col = rng_() % cols;
            T value = dist_.generate(rng_);
            triples.emplace_back(row, col, value);
        }
        return triples;
    }
};

template <class T>
void write_matrices_to_mm_file(const std::vector<TripleMatrix<T>>& matrices, size_t rows, size_t cols, double sparsity,
                               unsigned long seed, const std::string& dtype, const std::string& min_max) {
    std::ostringstream folder_name;
    folder_name << "#" << matrices.size() << "_" << rows << "x" << cols << "_" << sparsity * 100 << "%_" << seed 
                << "_" << dtype << "_" << min_max;
    std::filesystem::path full_path = std::filesystem::path("../data") / folder_name.str();
    std::filesystem::create_directories(full_path);

    for (size_t i = 0; i < matrices.size(); ++i) {
        std::ostringstream file_name;
        file_name << full_path.string() << "/matrix" << i + 1 << ".mtx";
        CSRMatrix<T>::to_mm_file(file_name.str(), matrices[i], rows, cols);
    }
    std::cout << "Matrices written to folder: " << full_path << "\n";
}

template <class T>
void run(size_t num_matrices, size_t rows, size_t cols, double sparsity, unsigned long seed, T min, T max) {
    MatrixGenerator<T> generator(seed, min, max);
    std::vector<TripleMatrix<T>> matrices;
    matrices.reserve(num_matrices);
    for (size_t i = 0; i < num_matrices; ++i) {
        auto matrix = generator.generate(rows, cols, sparsity);
        matrices.push_back(matrix);
    }
    write_matrices_to_mm_file(matrices, rows, cols, sparsity, seed, "uniform_int", std::to_string(min) + "_" + std::to_string(max));
}

int parse_and_run(int argc, char* argv[]) {
    // TODO: make min and max optional, because they are also optional for MatrixGenerator
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <#matrices> <rows> <cols> <sparsity> <seed> <int|double> <min> <max>\n";
        return -1;
    }
    const long num_matrices = std::stol(argv[1]);
    const long rows = std::stol(argv[2]);
    const long cols = std::stol(argv[3]);
    const double sparsity = std::stod(argv[4]);
    const long seed = std::stol(argv[5]);
    const std::string dtype = argv[6];

    // Validate inputs to ensure they are within valid ranges
    if (num_matrices <= 0 || rows <= 0 || cols <= 0 || sparsity < 0.0 || sparsity > 1.0 || seed < 0) {
        std::cerr << "Invalid argument.\n";
        return -1;
    }

    // Convert validated values to unsigned types
    const size_t num_matrices_u = static_cast<size_t>(num_matrices);
    const size_t rows_u = static_cast<size_t>(rows);
    const size_t cols_u = static_cast<size_t>(cols);
    const unsigned long seed_u = static_cast<unsigned long>(seed);
    
    if (dtype == "int") {
        int min = std::stoi(argv[7]);
        int max = std::stoi(argv[8]);
        run(num_matrices_u, rows_u, cols_u, sparsity, seed_u, min, max);
    } else if (dtype == "double") {
        double min = std::stod(argv[7]);
        double max = std::stod(argv[8]);
        run(num_matrices_u, rows_u, cols_u, sparsity, seed_u, min, max);
    } else {
        std::cerr << "Invalid argument.\n";
        return -1;
    }

    return 0;
}

#ifndef LIB_USAGE
int main(int argc, char* argv[]) {
    return parse_and_run(argc, argv);
}
#endif
