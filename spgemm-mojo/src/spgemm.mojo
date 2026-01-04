from testing import assert_equal
from csr import CSRMatrix

fn gustavson[N: Int](
        A: CSRMatrix,
        B: CSRMatrix,
    ) raises -> CSRMatrix:
    if A.cols != B.rows:
        raise Error("Incompatible matrix dimensions for multiplication. A.cols {} != B.rows {}".format(A.cols, B.rows))
    if N != B.cols:
        raise Error("Output matrix column size does not match. Expected {}, got {}".format(B.cols, N))

    var C = CSRMatrix(A.rows, N)
    var ip = 0
    var xb = List[Int](length=N, fill=Int.MAX)
    var x = List[Int](length=N, fill=Int())

    ref IA = A.row_ptr
    ref JA = A.col_idx
    ref A_val = A.values

    ref IB = B.row_ptr
    ref JB = B.col_idx
    ref B_val = B.values

    for i in range(len(IA) - 1):
        C.row_ptr[i] = ip
        var start_row_A = IA[i]
        var end_row_A = IA[i + 1]
        for jp in range(start_row_A, end_row_A):
            var j = JA[jp]
            var start_row_B = IB[j]
            var end_row_B = IB[j + 1]
            var A_val_jp = A_val[jp]
            for kp in range(start_row_B, end_row_B):
                var k = JB[kp]
                var B_val_kp = B_val[kp]
                if xb[k] != i:
                    C.col_idx.append(k)
                    ip += 1
                    xb[k] = i
                    x[k] = A_val_jp * B_val_kp
                else:
                    x[k] += A_val_jp * B_val_kp
        ref IC = C.row_ptr
        ref JC = C.col_idx
        for vp in range(IC[i], ip):
            var v = JC[vp]
            C.values.append(x[v])

    assert_equal(len(C.row_ptr), len(IA))
    C.row_ptr[len(IA) - 1] = ip
    return C^
