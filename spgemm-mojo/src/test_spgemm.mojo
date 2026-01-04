from testing import assert_equal, TestSuite
from csr import CSRMatrix, DenseMatrix
from spgemm import gustavson

comptime I = DenseMatrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
    )
comptime general = DenseMatrix([
        [2, 0, 2],
        [0, 0, 0],
        [2, 0, 2]]
    )

def test_empty():
    var empty_dense = DenseMatrix(List[List[Int]]())
    var a = CSRMatrix(empty_dense^)
    var c = gustavson[0](a, a)
    assert_equal(c.rows, 0)
    assert_equal(c.cols, 0)
    assert_equal(len(c.values), 0)

def test_single_element():
    var dense = DenseMatrix([[5]])
    var a = CSRMatrix(dense^)
    var c = gustavson[1](a, a)
    assert_equal(c.rows, 1)
    assert_equal(c.cols, 1)
    assert_equal(c.values, [25])

def test_identity():
    var dense = DenseMatrix([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    var a = CSRMatrix(dense^)
    var c = gustavson[3](a, a)
    assert_equal(c.rows, 3)
    assert_equal(c.cols, 3)
    assert_equal(c.values, [1, 1, 1])
    assert_equal(c.col_idx, [0, 1, 2])
    assert_equal(c.row_ptr, [0, 1, 2, 3])

def test_general():
    var a_dense = DenseMatrix([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 5]]
    )
    var b_dense = DenseMatrix([
        [7, 0, 0],
        [0, 8, 0],
        [9, 0, 10]]
    )
    var a = CSRMatrix(a_dense^)
    var b = CSRMatrix(b_dense^)
    var c = gustavson[3](a, b)
    assert_equal(c.rows, 3)
    assert_equal(c.cols, 3)
    assert_equal(c.values, [25, 20, 24, 73, 50])
    assert_equal(c.col_idx, [0, 2, 1, 0, 2])
    assert_equal(c.row_ptr, [0, 2, 3, 5])

def test_identity_chained():
    var g = materialize[T=DenseMatrix, value=I]()
    var a = CSRMatrix(g.copy())
    var b = CSRMatrix(g^)
    var c = gustavson[3](a, b)
    var d = gustavson[3](c, a)

    assert_equal(d.rows, 3)
    assert_equal(d.cols, 3)
    assert_equal(d.values, [1, 1, 1])
    assert_equal(d.col_idx, [0, 1, 2])
    assert_equal(d.row_ptr, [0, 1, 2, 3])

def test_general_chained():
    var dense = DenseMatrix([
        [2, 0, 2],
        [0, 2, 0],
        [2, 0, 2]]
    )
    var a = CSRMatrix(dense.copy())
    var b = CSRMatrix(dense^)
    var c = gustavson[3](a, b)

    assert_equal(c.rows, 3)
    assert_equal(c.cols, 3)
    assert_equal(c.values, [8, 8, 4, 8, 8])
    assert_equal(c.col_idx, [0, 2, 1, 0, 2])
    assert_equal(c.row_ptr, [0, 2, 3, 5])

    var d = gustavson[3](a, c)

    assert_equal(d.rows, 3)
    assert_equal(d.cols, 3)
    assert_equal(d.values, [32, 32, 8, 32, 32])
    assert_equal(d.col_idx, [0, 2, 1, 0, 2])
    assert_equal(d.row_ptr, [0, 2, 3, 5])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
