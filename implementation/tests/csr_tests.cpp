#include <gtest/gtest.h>
#include "common.h"
#include "csr.hpp"

// -+--+--+--+--+--+--+--+--+--+--+--+- Dense constructor -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(DenseConstructor, EmptyMatrix) {
    std::vector<std::vector<int>> dense_matrix = {};
    CSRMatrix<int> csr(dense_matrix);

    EXPECT_EQ(csr.values().size(), 0);
    EXPECT_EQ(csr.col_idx().size(), 0);
    EXPECT_EQ(csr.row_ptr().size(), 1);
    // sentinel value
    EXPECT_EQ(csr.row_ptr()[0], 0);

    EXPECT_EQ(csr.to_dense(), dense_matrix);
}

TEST(DenseConstructor, SingleRowMatrix) {
    std::vector<std::vector<int>> dense_matrix = {{0, 3, 0, 4}};
    CSRMatrix<int> csr(dense_matrix);

    EXPECT_EQ(csr.values(), (std::vector<int>{3, 4}));
    EXPECT_EQ(csr.values().size(), 2);
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{1, 3}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 2}));

    EXPECT_EQ(csr.to_dense(), dense_matrix);
}

TEST(DenseConstructor, SingleColumnMatrix) {
    std::vector<std::vector<int>> dense_matrix = {{1}, 
                                                 {0},
                                                 {2},
                                                 {0}};
    CSRMatrix<int> csr(dense_matrix);

    EXPECT_EQ(csr.values(), (std::vector<int>{1, 2}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{0, 0}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 1, 1, 2, 2}));

    EXPECT_EQ(csr.to_dense(), dense_matrix);
}

TEST(DenseConstructor, GeneralMatrix) {
    std::vector<std::vector<int>> dense_matrix = {
        {5, 0, 0, 0},
        {0, 8, 0, 0},
        {0, 0, 3, 0},
        {0, 6, 0, 0}
    };
    CSRMatrix<int> csr(dense_matrix);

    EXPECT_EQ(csr.values(), (std::vector<int>{5, 8, 3, 6}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{0, 1, 2, 1}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 1, 2, 3, 4}));

    EXPECT_EQ(csr.to_dense(), dense_matrix);
}

TEST(DenseConstructor, ZeroMatrix) {
    std::vector<std::vector<int>> dense_matrix = {
        {0, 0},
        {0, 0}
    };
    CSRMatrix<int> csr(dense_matrix);

    EXPECT_EQ(csr.values().size(), 0);
    EXPECT_EQ(csr.col_idx().size(), 0);
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 0, 0}));

    EXPECT_EQ(csr.to_dense(), dense_matrix);
}

// -+--+--+--+--+--+--+--+--+--+--+--+- Triple constructor -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(TripleConstructor, EmptyMatrix) {
    TripleMatrix<int> triples = {};
    CSRMatrix<int> csr(triples, 0, 0);

    EXPECT_EQ(csr.values().size(), 0);
    EXPECT_EQ(csr.col_idx().size(), 0);
    EXPECT_EQ(csr.row_ptr().size(), 1);
    // sentinel value
    EXPECT_EQ(csr.row_ptr()[0], 0);

    EXPECT_EQ(csr.to_triple(), triples);
}

TEST(TripleConstructor, SingleRowMatrix) {
    TripleMatrix<int> triples = {{0, 1, 3}, {0, 3, 4}};
    CSRMatrix<int> csr(triples, 1, 4);

    EXPECT_EQ(csr.values(), (std::vector<int>{3, 4}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{1, 3}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 2}));

    EXPECT_EQ(csr.to_triple(), triples);
}

TEST(TripleConstructor, SingleColumnMatrix) {
    TripleMatrix<int> triples = {{0, 0, 1}, {2, 0, 2}};
    CSRMatrix<int> csr(triples, 4, 1);

    EXPECT_EQ(csr.values(), (std::vector<int>{1, 2}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{0, 0}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 1, 1, 2, 2}));

    EXPECT_EQ(csr.to_triple(), triples);
}

TEST(TripleConstructor, GeneralMatrix) {
    TripleMatrix<int> triples = {{0, 0, 5}, {1, 1, 8}, {2, 2, 3}, {3, 1, 6}};
    CSRMatrix<int> csr(triples, 4, 4);

    EXPECT_EQ(csr.values(), (std::vector<int>{5, 8, 3, 6}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{0, 1, 2, 1}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 1, 2, 3, 4}));

    EXPECT_EQ(csr.to_triple(), triples);
}

TEST(TripleConstructor, ZeroMatrix) {
    TripleMatrix<int> triples = {};
    CSRMatrix<int> csr(triples, 2, 2);

    EXPECT_EQ(csr.values().size(), 0);
    EXPECT_EQ(csr.col_idx().size(), 0);
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 0, 0}));

    EXPECT_EQ(csr.to_triple(), triples);
}

TEST(TripleConstructor, NonSortedTriples) {
    TripleMatrix<int> triples = {{0, 1, 3}, {0, 3, 4}, {0, 0, 5}};
    CSRMatrix<int> csr(triples, 1, 4);

    EXPECT_EQ(csr.values(), (std::vector<int>{5, 3, 4}));
    EXPECT_EQ(csr.col_idx(), (std::vector<size_t>{0, 1, 3}));
    EXPECT_EQ(csr.row_ptr(), (std::vector<size_t>{0, 3}));

    TripleMatrix<int> sorted_triples = {{0, 0, 5}, {0, 1, 3}, {0, 3, 4}};
    EXPECT_EQ(csr.to_triple(), sorted_triples);
}

// -+--+--+--+--+--+--+--+--+--+--+--+- Test to + from matrix market file -+--+--+--+--+--+--+--+--+--+--+--+- //
TEST(MMFile, StoreGeneralMatrix) {
    auto filename = "test_store_general.mtx";

    TripleMatrix<int> triples = {{0, 0, 1}, {0, 1, 4}, {1, 1, 2}, {2, 2, 3}};
    CSRMatrix<int>::to_mm_file(filename, triples, 4, 4);

    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "%%MatrixMarket matrix coordinate integer general");

    // rows cols nnz
    std::getline(file, line);
    EXPECT_EQ(line, "4 4 4");

    std::getline(file, line);
    EXPECT_EQ(line, "1 1 1");
    std::getline(file, line);
    EXPECT_EQ(line, "1 2 4");
    std::getline(file, line);
    EXPECT_EQ(line, "2 2 2");
    std::getline(file, line);
    EXPECT_EQ(line, "3 3 3");

    std::remove(filename);
}
TEST(MMFile, StoreSymmetricMatrix) {
    auto filename = "test_store_symmetric.mtx";

    TripleMatrix<int> triples = {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}, {3, 3, 4}};
    CSRMatrix<int>::to_mm_file(filename, triples, 4, 4);

    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "%%MatrixMarket matrix coordinate integer symmetric");

    // rows cols nnz
    std::getline(file, line);
    EXPECT_EQ(line, "4 4 4");

    std::getline(file, line);
    EXPECT_EQ(line, "1 1 1");
    std::getline(file, line);
    EXPECT_EQ(line, "2 2 2");
    std::getline(file, line);
    EXPECT_EQ(line, "3 3 3");
    std::getline(file, line);
    EXPECT_EQ(line, "4 4 4");

    std::remove(filename);
}

TEST(MMFile, GeneralMatrix) {
    std::vector<std::vector<int>> dense_matrix = {
        {1, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 6, 3, 0},
        {0, 0, 0, 4}
    };
    CSRMatrix<int> csr(dense_matrix);
    auto filename = "test_general.mtx";
    csr.to_mm_file(filename);
    EXPECT_EQ(csr.to_triple().size(), 5);
    CSRMatrix<int> csr_from_mm = CSRMatrix<int>::from_mm_file(filename);
    EXPECT_EQ(csr, csr_from_mm);

    std::remove(filename);
}

TEST(MMFile, SymmetricMatrix) {
    std::vector<std::vector<double>> dense_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 2.0, 6.0, 0.0},
        {0.0, 6.0, 3.0, 0.0},
        {0.0, 0.0, 0.0, 4.0}
    };
    CSRMatrix<double> csr(dense_matrix);
    auto filename = "test_symmetric.mtx";
    csr.to_mm_file(filename);

    EXPECT_EQ(csr.to_triple().size(), 6);
    CSRMatrix<double> csr_from_mm = CSRMatrix<double>::from_mm_file(filename);
    EXPECT_EQ(csr, csr_from_mm);

    std::remove(filename);
}

TEST(MMFile, PatternMatrix) {
    std::ofstream file("test_pattern.mtx");
    file << "%%MatrixMarket matrix coordinate pattern general\n";
    file << "3 3 3\n";
    file << "1 1\n";
    file << "2 2\n";
    file << "3 3\n";
    file.close();

    CSRMatrix<int> csr = CSRMatrix<int>::from_mm_file("test_pattern.mtx");

    DenseMatrix<int> expected = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    EXPECT_EQ(csr.to_dense(), expected);
}

TEST(MMFile, ScientificNotation) {
    std::ofstream file("test_scientific_notation.mtx");
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "3 3 3\n";
    file << "1 1 1.5e-1\n";
    file << "2 2 2.5e-1\n";
    file << "3 3 3.5e-1\n";
    file.close();

    CSRMatrix<double> csr = CSRMatrix<double>::from_mm_file("test_scientific_notation.mtx");
    EXPECT_EQ(csr.values(), (std::vector<double>{1.5e-1, 2.5e-1, 3.5e-1}));

    std::remove("test_scientific_notation.mtx");
}

TEST(MMFile, InvalidDataType) {
    std::ofstream file("test_invalid_data.mtx");
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "3 3 3\n";
    file << "1 1 1.5\n";
    file << "2 2 2.5\n";
    file << "3 3 3.5\n";
    file.close();

    EXPECT_THROW(CSRMatrix<int>::from_mm_file("test_invalid_data.mtx"), std::runtime_error);

    std::remove("test_invalid_data.mtx");
}

TEST(MMFile, IncorrectFormat) {
    std::ofstream file("test_invalid_format.mtx");
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "3 3\n";  // Missing nnz count
    file.close();

    EXPECT_THROW(CSRMatrix<int>::from_mm_file("test_invalid_format.mtx"), std::runtime_error);

    std::remove("test_invalid_format.mtx");
}
