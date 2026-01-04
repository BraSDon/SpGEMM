from benchmark import run, Unit, Report
from benchmark.compiler import keep
from csr import CSRMatrix, DenseMatrix
from spgemm import gustavson
from matrix_generator import MatrixGenerator

fn main() raises:
    var generator = MatrixGenerator(42)
    comptime sizes = [64, 128, 256, 512, 1024, 2048]
    comptime densities = [0.01, 0.05, 0.1, 0.2]
    @parameter
    for size in sizes:
        @parameter
        for density in densities:
            var mat_a = CSRMatrix(generator.generate(size, size, density))
            var mat_b = CSRMatrix(generator.generate(size, size, density))
            print("Benchmarking size={} density={}".format(size, density))
            bench[size](mat_a, mat_b)
            print("")

fn bench[N: Int](a: CSRMatrix, b: CSRMatrix) raises:
    @parameter
    fn inner() raises:
        _ = keep(gustavson[N](a, b))
    var report = run[inner](max_iters=100)
    print("Mean: {} ms".format(report.mean(unit=Unit.ms).__round__(4)))
