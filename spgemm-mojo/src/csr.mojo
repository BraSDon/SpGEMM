@fieldwise_init
struct TripleEntry(Copyable, Comparable):
    var row: Int
    var col: Int
    var value: Int

    @implicit
    fn __init__(out self, tuple: Tuple[Int, Int, Int]):
        self.row = tuple[0]
        self.col = tuple[1]
        self.value = tuple[2]

    fn __eq__(self, other: TripleEntry) -> Bool:
        return self.row == other.row and self.col == other.col and self.value == other.value

    fn __lt__(self, other: TripleEntry) -> Bool:
        if self.row != other.row:
            return self.row < other.row
        if self.col != other.col:
            return self.col < other.col
        return False

struct TripleMatrix(Movable):
    var rows: Int
    var cols: Int
    var entries: List[TripleEntry]

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.entries = List[TripleEntry]()

    fn __init__(out self, rows: Int, cols: Int, var entries: List[TripleEntry]):
        self.rows = rows
        self.cols = cols
        self.entries = entries^

    fn add_entry(self, row: Int, col: Int, value: Int):
        self.entries.append((row, col, value))

struct DenseMatrix(Copyable, Stringable):
    var data: List[List[Int]]

    fn __init__(out self, var data: List[List[Int]]):
        self.data = data^

    fn num_rows(self) -> Int:
        return len(self.data)

    fn num_cols(self) -> Int:
        if len(self.data) == 0:
            return 0
        return len(self.data[0])

    fn __getitem__(self, row: Int, col: Int) -> Int:
        return self.data[row][col]

    fn __bool__(self) -> Bool:
        return len(self.data) > 0 and len(self.data[0]) > 0

    fn __str__(self) -> String:
        var res = String()
        for i, r in enumerate(self.data):
            for j, entry in enumerate(r):
                res += String(entry)
                if j < len(r) - 1:
                    res += ", "
            if i < len(self.data) - 1:
                res += "\n"
        return res

struct CSRMatrix(Movable, Defaultable, Stringable):
    var rows: Int
    var cols: Int
    var values: List[Int]
    var col_idx: List[Int]
    var row_ptr: List[Int]

    fn __init__(out self):
        self.rows = 0
        self.cols = 0
        self.values = List[Int]()
        self.col_idx = List[Int]()
        self.row_ptr = List[Int](length=self.rows + 1, fill=0)

    # Known nnz (number of non-zero elements)
    fn __init__(out self, rows: Int, cols: Int, nnz: Int):
        self.rows = rows
        self.cols = cols
        self.values = List[Int](capacity=nnz)
        self.col_idx = List[Int](capacity=nnz)
        self.row_ptr = List[Int](length=self.rows + 1, fill=0)

    # Unknown nnz
    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.values = List[Int]()
        self.col_idx = List[Int]()
        self.row_ptr = List[Int](length=self.rows + 1, fill=0)

    fn __init__(out self, var matrix: DenseMatrix):
        # Handle empty matrix
        if not matrix:
            return self.__init__(0, 0)

        self.rows = matrix.num_rows()
        self.cols = matrix.num_cols()
        self.values = List[Int]()
        self.col_idx = List[Int]()
        self.row_ptr = List[Int](length=self.rows + 1, fill=0)

        for row in range(self.rows):
            for col in range(self.cols):
                var val = matrix[row, col]
                if val != 0:
                    self.values.append(val)
                    self.col_idx.append(col)

            self.row_ptr[row + 1] = len(self.values)

    fn __init__(out self, var matrix: TripleMatrix):
        self.rows = matrix.rows
        self.cols = matrix.cols
        self.values = List[Int](capacity=len(matrix.entries))
        self.col_idx = List[Int](capacity=len(matrix.entries))
        self.row_ptr = List[Int](length=self.rows + 1, fill=0)

        var entries = matrix.entries.copy()
        sort(entries)

        for entry in entries:
            self.values.append(entry.value)
            self.col_idx.append(entry.col)
            self.row_ptr[entry.row + 1] += 1

        for row in range(self.rows):
            self.row_ptr[row + 1] += self.row_ptr[row]


    fn __str__(self) -> String:
        var res = String()
        res += "CSRMatrix(rows={}, cols={}, nnz={})\n".format(self.rows, self.cols, len(self.values))
        res += " Values: "
        for v in self.values:
            res += "{} ".format(v)
        res += "\n Col Indices: "
        for c in self.col_idx:
            res += "{} ".format(c)
        res += "\n Row Pointers: "
        for r in self.row_ptr:
            res += "{} ".format(r)
        return res

    fn to_dense(self) -> DenseMatrix:
        var dense = List[List[Int]](length=self.rows, fill=List[Int](length=self.cols, fill=0))
        for row in range(self.rows):
            for idx in range(self.row_ptr[row], self.row_ptr[row + 1]):
                print("Processing row {}, idx {}".format(row, idx))
                var col = self.col_idx[idx]

                print("Setting dense[{}][{}] = {}".format(row, col, self.values[idx]))
                dense[row][col] = self.values[idx]

        return DenseMatrix(dense^)



