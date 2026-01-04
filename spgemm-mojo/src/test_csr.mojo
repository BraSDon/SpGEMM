from testing import assert_equal, TestSuite
from csr import CSRMatrix, DenseMatrix, TripleMatrix, TripleEntry

def test_init_from_empty_dense():
    var empty_dense = DenseMatrix(List[List[Int]]())
    var csr = CSRMatrix(empty_dense^)
    assert_equal(csr.rows, 0)
    assert_equal(csr.cols, 0)
    assert_equal(len(csr.values), 0)
    assert_equal(len(csr.col_idx), 0)
    assert_equal(len(csr.row_ptr), 1)
    # sentinel value
    assert_equal(csr.row_ptr[0], 0)

def test_init_from_single_row_dense():
    var dense = DenseMatrix([[0, 3, 0, 4]])
    var csr = CSRMatrix(dense^)

    assert_equal(csr.rows, 1)
    assert_equal(csr.cols, 4)
    assert_equal(csr.values, [3, 4])
    assert_equal(csr.col_idx, [1, 3])
    assert_equal(csr.row_ptr, [0, 2])

def test_init_from_single_column_dense():
    # 1, 0, 2, 0
    var dense = DenseMatrix([[1], [0], [2], [0]])
    var csr = CSRMatrix(dense^)

    assert_equal(csr.rows, 4)
    assert_equal(csr.cols, 1)
    assert_equal(csr.values, [1, 2])
    assert_equal(csr.col_idx, [0, 0])
    assert_equal(csr.row_ptr, [0, 1, 1, 2, 2])

def test_init_from_dense():
    var dense = DenseMatrix([[5, 0, 0, 0],
                                  [0, 8, 0, 0],
                                  [0, 0, 3, 0],
                                  [0, 6, 0, 0]])
    var csr = CSRMatrix(dense^)

    assert_equal(csr.rows, 4)
    assert_equal(csr.cols, 4)
    assert_equal(csr.values, [5, 8, 3, 6])
    assert_equal(csr.col_idx, [0, 1, 2, 1])
    assert_equal(csr.row_ptr, [0, 1, 2, 3, 4])

def test_init_from_triple():
    var entries = [
        TripleEntry(0, 0, 10),
        TripleEntry(0, 2, 20),
        TripleEntry(2, 0, 40),
        TripleEntry(2, 2, 50),
        TripleEntry(1, 1, 30)
    ]
    var triple = TripleMatrix(3, 3, entries^)
    var csr = CSRMatrix(triple^)

    assert_equal(csr.rows, 3)
    assert_equal(csr.cols, 3)
    assert_equal(csr.values, [10, 20, 30, 40, 50])
    assert_equal(csr.col_idx, [0, 2, 1, 0, 2])
    assert_equal(csr.row_ptr, [0, 2, 3, 5])

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
