from random import seed, random_si64
from csr import CSRMatrix, TripleMatrix, TripleEntry
from collections import Set

struct MatrixGenerator:
    var seed: Int
    var min: Int
    var max: Int

    fn __init__(out self, seed_val: Int, min: Int = -5, max: Int = 5):
        self.seed = seed_val
        self.min = min
        self.max = max
        seed(self.seed)

    fn _randint(self, min: Int, max: Int) -> Int:
        return Int(random_si64(Int64(min), Int64(max)))

    fn _randint(self) -> Int:
        return self._randint(self.min, self.max)

    fn _nonzero(self) -> Int:
        while True:
            var val = self._randint()
            if val != 0:
                return val

    fn generate(self, rows: Int, cols: Int, density: Float64) -> TripleMatrix:
        var nnz_target = Int(Float64(rows * cols) * density)
        var entries = List[TripleEntry](capacity=nnz_target)
        var seen = Set[Int]() 

        while len(entries) < nnz_target:
            var row = self._randint(0, rows - 1)
            var col = self._randint(0, cols - 1)
            var linear_idx = row * cols + col

            if linear_idx not in seen:
                seen.add(linear_idx)
                var val = self._nonzero()
                entries.append((row, col, val))

        return TripleMatrix(rows, cols, entries^)^


fn main():
    var generator = MatrixGenerator(42)
    var matrix = generator.generate(4, 4, 0.5)
    var csr = CSRMatrix(matrix^)
    print(String(csr.to_dense()))
