#include <filesystem>
#include <gtest/gtest.h>
#include "matrix_generator.cpp"

// -+--+--+--+--+--+--+--+--+--+--+--+- DistributionHelper -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(DistributionHelper, IntDistribution) {
    DistributionHelper<int> helper(1, 10);
    std::mt19937_64 engine(42);

    for (int i = 0; i < 100; ++i) {
        int value = helper.generate(engine);
        EXPECT_GE(value, 1);
        EXPECT_LE(value, 10);
    }
}

TEST(DistributionHelper, DoubleDistribution) {
    DistributionHelper<double> helper(-10.0, 10.0);
    std::mt19937_64 engine(42);

    for (int i = 0; i < 100; ++i) {
        double value = helper.generate(engine);
        EXPECT_GE(value, -10.0);
        EXPECT_LE(value, 10.0);
    }
}

// -+--+--+--+--+--+--+--+--+--+--+--+- MatrixGenerator -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(MatrixGenerator, Bounds) {
    size_t rows = 5, cols = 5;
    double sparsity = 0.2;
    unsigned long seed = 42;
    int min = 1, max = 10;

    MatrixGenerator<int> generator(seed, min, max);
    std::mt19937_64 engine(seed);
    TripleMatrix<int> matrix = generator.generate(rows, cols, sparsity);

    for (const auto& triple : matrix) {
        EXPECT_GE(std::get<2>(triple), 1);
        EXPECT_LE(std::get<2>(triple), 10);
    }

    // Double bounds
    MatrixGenerator<double> generator_double(seed, -10.0, 10.0);
    TripleMatrix<double> matrix_double = generator_double.generate(rows, cols, sparsity);

    for (const auto& triple : matrix_double) {
        EXPECT_GE(std::get<2>(triple), -10.0);
        EXPECT_LE(std::get<2>(triple), 10.0);
    }
}

TEST(MatrixGenerator, NNZ) {
    size_t rows = 5, cols = 5;
    double sparsity = 0.2;
    unsigned long seed = 42;
    int min = 1, max = 10;

    MatrixGenerator<int> generator(seed, min, max);
    TripleMatrix<int> matrix = generator.generate(rows, cols, sparsity);

    size_t expected_nnz = static_cast<size_t>(static_cast<double>(rows * cols) * (1 - sparsity));
    EXPECT_EQ(matrix.size(), expected_nnz);
}

TEST(MatrixGenerator, SameSeedSameMatrix) {
    size_t rows = 5, cols = 5;
    double sparsity = 0.2;
    unsigned long seed = 42;
    int min = 1, max = 10;

    MatrixGenerator<int> generator(seed, min, max);
    TripleMatrix<int> matrix1 = generator.generate(rows, cols, sparsity);

    // reset generator
    generator.reset_seed(seed);
    TripleMatrix<int> matrix2 = generator.generate(rows, cols, sparsity);

    EXPECT_EQ(matrix1, matrix2);
}

TEST(MatrixGenerator, Dimensions) {
    // Fat short, tall thin, square
    std::vector<std::pair<size_t, size_t>> dimensions = { {2, 10}, {10, 2}, {5, 5} };
    double sparsity = 0.9;
    unsigned long seed = 42;
    int min = 1, max = 10;

    for (const auto& [rows, cols] : dimensions) {
        MatrixGenerator<int> generator(seed, min, max);
        TripleMatrix<int> matrix = generator.generate(rows, cols, sparsity);

        for (const auto& triple : matrix) {
            EXPECT_GE(std::get<0>(triple), 0);
            EXPECT_LT(std::get<0>(triple), rows);
            EXPECT_GE(std::get<1>(triple), 0);
            EXPECT_LT(std::get<1>(triple), cols);
        }
    }
}

// -+--+--+--+--+--+--+--+--+--+--+--+- File creation -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(FileOutput, CreateDirectoriesAndFiles) {
    size_t rows = 3, cols = 3;
    double sparsity = 0.5;
    unsigned long seed = 42;

    std::vector<TripleMatrix<int>> matrices_int = { { {0, 0, 1}, {1, 2, 3} } };
    std::vector<TripleMatrix<double>> matrices_double = { { {0.0, 0.0, 1.0}, {1.0, 2.0, 3.0} } };

    write_matrices_to_mm_file(matrices_int, rows, cols, sparsity, seed, "int", "1_10");
    std::string folder_name = "../data/#1_3x3_50%_42_int_1_10";
    std::filesystem::path file_path = std::filesystem::path(folder_name) / "matrix1.mtx";
    EXPECT_TRUE(std::filesystem::exists(file_path));

    write_matrices_to_mm_file(matrices_double, rows, cols, sparsity, seed, "double", "0.0_3.0");
    folder_name = "../data/#1_3x3_50%_42_double_0.0_3.0";
    file_path = std::filesystem::path(folder_name) / "matrix1.mtx";
    EXPECT_TRUE(std::filesystem::exists(file_path));

    // Cleanup
    std::filesystem::remove_all(folder_name);
}
