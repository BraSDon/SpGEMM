#include <gtest/gtest.h>
#include <Eigen/Sparse>
#include <vector>

#include "common.h"
#include "spgemm.hpp"
#include "matrix_generator.cpp"
#include "csr.hpp"

// TODO: 
// - test for numerical issues with floats/doubles

// Base test fixture for parameterized SpGEMM tests
template <typename Dtype>
class SpGEMM : public ::testing::Test {};

TYPED_TEST_SUITE_P(SpGEMM);

TYPED_TEST_P(SpGEMM, EdgeCases) {
    using Dtype = TypeParam;

    // Empty matrices
    CSRMatrix<Dtype> A_empty;
    CSRMatrix<Dtype> B_empty;
    auto C_empty = gustavson<Dtype, 0>(A_empty, B_empty).to_triple();
    EXPECT_TRUE(C_empty.empty());

    // Single element matrices
    CSRMatrix<Dtype> A_single({{0, 0, static_cast<Dtype>(1)}}, 1, 1);
    CSRMatrix<Dtype> B_single({{0, 0, static_cast<Dtype>(2)}}, 1, 1);
    auto C_single = gustavson<Dtype, 1>(A_single, B_single).to_triple();
    EXPECT_EQ(C_single.size(), 1);

    // extract value from C_single, which is vector<tuple<int, int, Dtype>>
    auto [i, j, value] = C_single[0];
    if constexpr (std::is_floating_point<Dtype>::value) {
        EXPECT_NEAR(value, static_cast<Dtype>(2), 1e-6);
    } else {
        EXPECT_EQ(value, static_cast<Dtype>(2));
    }
}

TYPED_TEST_P(SpGEMM, GeneralMatrix) {
    using Dtype = TypeParam;

    // diagonal with 2s
    DenseMatrix<Dtype> dense = {
        {2, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 2, 0},
        {0, 0, 0, 2}
    };
    CSRMatrix<Dtype> A(dense);
    auto C = gustavson<Dtype, 4>(A, A).to_triple();

    EXPECT_EQ(C.size(), 4);
    EXPECT_EQ(C, TripleMatrix<Dtype>({
        {0, 0, static_cast<Dtype>(4)},
        {1, 1, static_cast<Dtype>(4)},
        {2, 2, static_cast<Dtype>(4)},
        {3, 3, static_cast<Dtype>(4)}
    }));
}

// Test with 100 randomly generated matrices
TYPED_TEST_P(SpGEMM, RandomMatrices) {
    using Dtype = TypeParam;

    std::vector<TripleMatrix<Dtype>> matrices;
    matrices.reserve(100);

    MatrixGenerator<Dtype> generator(42, -10, 10);

    for (int i = 0; i < 100; ++i) {
        auto matrix = generator.generate(4, 4, 0.5);
        matrices.push_back(matrix);
    }

    for (const auto& matrix : matrices) {
        CSRMatrix<Dtype> A(matrix, 4, 4);
        auto C = gustavson<Dtype, 4>(A, A).to_dense();

        Eigen::SparseMatrix<Dtype> A_eigen(4, 4);
        auto triplets = to_eigen_triplets(matrix);
        A_eigen.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SparseMatrix<Dtype> C_eigen = A_eigen * A_eigen;

        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                const auto i_li = static_cast<long int>(i);
                const auto j_li = static_cast<long int>(j);
                auto value = C[i][j];
                if constexpr (std::is_floating_point<Dtype>::value) {
                    EXPECT_NEAR(value, C_eigen.coeff(i_li, j_li), 1e-6) << "Mismatch at (" << i << ", " << j << ")";
                } else {
                    EXPECT_EQ(value, C_eigen.coeff(i_li, j_li)) << "Mismatch at (" << i << ", " << j << ")";
                }
            }
        }
    }
}

TYPED_TEST_P(SpGEMM, ParallelEdgeCases) {
    using Dtype = TypeParam;

    // Empty matrices
    CSRMatrix<Dtype> A_empty;
    CSRMatrix<Dtype> B_empty;
    auto C_empty = gustavson_parallel<Dtype, 0>(A_empty, B_empty).to_triple();
    EXPECT_TRUE(C_empty.empty());

    // Single element matrices
    CSRMatrix<Dtype> A_single({{0, 0, static_cast<Dtype>(1)}}, 1, 1);
    CSRMatrix<Dtype> B_single({{0, 0, static_cast<Dtype>(2)}}, 1, 1);
    auto C_single = gustavson_parallel<Dtype, 1>(A_single, B_single).to_triple();
    EXPECT_EQ(C_single.size(), 1);

    // extract value from C_single, which is vector<tuple<int, int, Dtype>>
    auto [i, j, value] = C_single[0];
    if constexpr (std::is_floating_point<Dtype>::value) {
        EXPECT_NEAR(value, static_cast<Dtype>(2), 1e-6);
    } else {
        EXPECT_EQ(value, static_cast<Dtype>(2));
    }
}

// Test with a known matrix, easier to debug.
TYPED_TEST_P(SpGEMM, ParallelGeneralMatrix) {
    using Dtype = TypeParam;

    // diagonal with 2s
    DenseMatrix<Dtype> dense = {
        {2, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 2, 0},
        {0, 0, 0, 2}
    };
    CSRMatrix<Dtype> A(dense);
    auto C = gustavson_parallel<Dtype, 4>(A, A).to_dense();

    Eigen::SparseMatrix<Dtype> A_eigen(4, 4);
    auto triplets = to_eigen_triplets(A.to_triple());
    A_eigen.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<Dtype> C_eigen = A_eigen * A_eigen;

    EXPECT_EQ(C.size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const auto i_li = static_cast<long int>(i);
            const auto j_li = static_cast<long int>(j);
            auto value = C[i][j];
            if constexpr (std::is_floating_point<Dtype>::value) {
                EXPECT_NEAR(value, C_eigen.coeff(i_li, j_li), 1e-6) << "Mismatch at (" << i << ", " << j << ")";
            } else {
                EXPECT_EQ(value, C_eigen.coeff(i_li, j_li)) << "Mismatch at (" << i << ", " << j << ")";
            }
        }
    }
}

// Test with 100 randomly generated matrices
TYPED_TEST_P(SpGEMM, ParallelRandomMatrices) {
    using Dtype = TypeParam;

    std::vector<TripleMatrix<Dtype>> matrices;
    matrices.reserve(100);

    MatrixGenerator<Dtype> generator(42, -10, 10);
    constexpr size_t dim = 32;

    for (int i = 0; i < 100; ++i) {
        auto matrix = generator.generate(dim, dim, 0.5);
        matrices.push_back(matrix);
    }

    for (const auto& matrix : matrices) {
        CSRMatrix<Dtype> A(matrix, dim, dim);
        auto C = gustavson_parallel<Dtype, dim>(A, A).to_dense();

        Eigen::SparseMatrix<Dtype> A_eigen(dim, dim);
        auto triplets = to_eigen_triplets(matrix);
        A_eigen.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SparseMatrix<Dtype> C_eigen = A_eigen * A_eigen;

        for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    const auto i_li = static_cast<long int>(i);
                    const auto j_li = static_cast<long int>(j);
                    auto value = C[i][j];
                    if constexpr (std::is_floating_point<Dtype>::value) {
                        EXPECT_NEAR(value, C_eigen.coeff(i_li, j_li), 1e-6) << "Mismatch at (" << i << ", " << j << ")";
                    } else {
                        EXPECT_EQ(value, C_eigen.coeff(i_li, j_li)) << "Mismatch at (" << i << ", " << j << ")";
                    }
                }
            }
    }
}

// Register all parameterized tests
REGISTER_TYPED_TEST_SUITE_P(
    SpGEMM,
    EdgeCases,
    GeneralMatrix,
    RandomMatrices,
    ParallelEdgeCases,
    ParallelGeneralMatrix,
    ParallelRandomMatrices
);

// Define the types to test
using DtypesToTest = ::testing::Types<
    double,
    int
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    SpGEMMTests,
    SpGEMM,
    DtypesToTest
);
