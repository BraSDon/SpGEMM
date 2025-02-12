#include <benchmark/benchmark.h>

#include "common.h"
#include "csr.hpp"
#include "matrix_generator.cpp"
#include "spgemm.hpp"

// TODO:
// - Benchmark with a set of representative matrices (e.g. Florida dataset, same as MA)
template <class T, size_t N>
void BM_Gustavson_random(benchmark::State &state) {
  const double sparsity = static_cast<double>(state.range(0)) / 100.0;
  CSRMatrix<T> A;
  MatrixGenerator<T> generator(42);
  A = CSRMatrix<T>(generator.generate(N, N, sparsity), N, N);

  for (auto _ : state) {
    benchmark::DoNotOptimize(gustavson<T, N>(A, A));
    benchmark::ClobberMemory();
  }
}

template <class T, size_t N, size_t NUM_THREADS = 8>
void BM_Parallel_random(benchmark::State &state) {
  const double sparsity = static_cast<double>(state.range(0)) / 100.0;
  CSRMatrix<T> A;
  MatrixGenerator<T> generator(42);
  A = CSRMatrix<T>(generator.generate(N, N, sparsity), N, N);

  for (auto _ : state) {
    benchmark::DoNotOptimize(gustavson_parallel<T, N, NUM_THREADS>(A, A));
    benchmark::ClobberMemory();
  }
}

template <class T, long int N>
void BM_Eigen_random(benchmark::State &state) {
  const double sparsity = static_cast<double>(state.range(0)) / 100.0;
  MatrixGenerator<T> generator(42);
  auto triplets = to_eigen_triplets(generator.generate(N, N, sparsity));
  Eigen::SparseMatrix<T> A(N, N);
  A.setFromTriplets(triplets.begin(), triplets.end());

  for (auto _ : state) {
    Eigen::SparseMatrix<T> C = A * A;
    benchmark::ClobberMemory();
  }
}

template <class T, size_t N>
void BM_Gustavson(benchmark::State &state, std::string path) {
  const auto full_path = "../../data/" + path;
  CSRMatrix<T> matrix = CSRMatrix<T>::from_mm_file(full_path);
  for (auto _ : state) {
    benchmark::DoNotOptimize(gustavson<T, N>(matrix, matrix));
    benchmark::ClobberMemory();
  }
}

template <class T, size_t N, size_t NUM_THREADS = 8>
void BM_Parallel(benchmark::State &state, std::string path) {
  const auto full_path = "../../data/" + path;
  CSRMatrix<T> matrix = CSRMatrix<T>::from_mm_file(full_path);
  for (auto _ : state) {
    benchmark::DoNotOptimize(gustavson_parallel<T, N, NUM_THREADS>(matrix, matrix));
    benchmark::ClobberMemory();
  }
}

template <class T, long int N>
void BM_Eigen(benchmark::State &state, std::string path) {
  const auto full_path = "../../data/" + path;
  CSRMatrix<T> matrix = CSRMatrix<T>::from_mm_file(full_path);
  Eigen::SparseMatrix<T> A = matrix.to_eigen();
  for (auto _ : state) {
    Eigen::SparseMatrix<T> C = A * A;
    benchmark::ClobberMemory();
  }
}

// nemeth07
void BM_Gustavson_Nemeth(benchmark::State &state) {
  BM_Gustavson<double, 9506>(state, "nemeth07/nemeth07.mtx");
}
void BM_Parallel_Nemeth(benchmark::State &state) {
  BM_Parallel<double, 9506>(state, "nemeth07/nemeth07.mtx");
}
void BM_Eigen_Nemeth(benchmark::State &state) {
  BM_Eigen<double, 9506>(state, "nemeth07/nemeth07.mtx");
}

// lhr71c
void BM_Gustavson_Lhr71c(benchmark::State &state) {
  BM_Gustavson<double, 70304>(state, "lhr71c/lhr71c.mtx");
}
void BM_Parallel_Lhr71c(benchmark::State &state) {
  BM_Parallel<double, 70304>(state, "lhr71c/lhr71c.mtx");
}
void BM_Eigen_Lhr71c(benchmark::State &state) {
  BM_Eigen<double, 70304>(state, "lhr71c/lhr71c.mtx");
}

// c-71
void BM_Gustavson_C71(benchmark::State &state) {
  BM_Gustavson<double, 76638>(state, "c-71/c-71.mtx");
}
void BM_Parallel_C71(benchmark::State &state) {
  BM_Parallel<double, 76638>(state, "c-71/c-71.mtx");
}
void BM_Eigen_C71(benchmark::State &state) {
  BM_Eigen<double, 76638>(state, "c-71/c-71.mtx");
}

// preferentialAttachment
void BM_Gustavson_preferentialAttachment(benchmark::State &state) {
  BM_Gustavson<int, 100000>(state, "preferentialAttachment/preferentialAttachment.mtx");
}
void BM_Parallel_preferentialAttachment(benchmark::State &state) {
  BM_Parallel<int, 100000>(state, "preferentialAttachment/preferentialAttachment.mtx");
}
void BM_Eigen_preferentialAttachment(benchmark::State &state) {
  BM_Eigen<int, 100000>(state, "preferentialAttachment/preferentialAttachment.mtx");
}

// consph
void BM_Gustavson_consph(benchmark::State &state) {
  BM_Gustavson<double, 83334>(state, "consph/consph.mtx");
}
void BM_Parallel_consph(benchmark::State &state) {
  BM_Parallel<double, 83334>(state, "consph/consph.mtx");
}
void BM_Eigen_consph(benchmark::State &state) {
  BM_Eigen<double, 83334>(state, "consph/consph.mtx");
}

// rgg_n_2_22_s0
void BM_Gustavson_rgg_n_2_22_s0(benchmark::State &state) {
  BM_Gustavson<int, 4194304>(state, "rgg_n_2_22_s0/rgg_n_2_22_s0.mtx");
}
void BM_Parallel_rgg_n_2_22_s0(benchmark::State &state) {
  BM_Parallel<int, 4194304>(state, "rgg_n_2_22_s0/rgg_n_2_22_s0.mtx");
}
void BM_Eigen_rgg_n_2_22_s0(benchmark::State &state) {
  BM_Eigen<int, 4194304>(state, "rgg_n_2_22_s0/rgg_n_2_22_s0.mtx");
}

// rajat31
void BM_Gustavson_rajat31(benchmark::State &state) {
  BM_Gustavson<double, 4690002>(state, "rajat31/rajat31.mtx");
}
void BM_Parallel_rajat31(benchmark::State &state) {
  BM_Parallel<double, 4690002>(state, "rajat31/rajat31.mtx");
}
void BM_Eigen_rajat31(benchmark::State &state) {
  BM_Eigen<double, 4690002>(state, "rajat31/rajat31.mtx");
}

// M6
void BM_Gustavson_M6(benchmark::State &state) {
  BM_Gustavson<int, 3501776>(state, "M6/M6.mtx");
}
void BM_Parallel_M6(benchmark::State &state) {
  BM_Parallel<int, 3501776>(state, "M6/M6.mtx");
}
void BM_Eigen_M6(benchmark::State &state) {
  BM_Eigen<int, 3501776>(state, "M6/M6.mtx");
}

// ASIC_680ks
void BM_Gustavson_ASIC_680ks(benchmark::State &state) {
  BM_Gustavson<double, 682712>(state, "ASIC_680ks/ASIC_680ks.mtx");
}
void BM_Parallel_ASIC_680ks(benchmark::State &state) {
  BM_Parallel<double, 682712>(state, "ASIC_680ks/ASIC_680ks.mtx");
}
void BM_Eigen_ASIC_680ks(benchmark::State &state) {
  BM_Eigen<double, 682712>(state, "ASIC_680ks/ASIC_680ks.mtx");
}

auto repeats = 4;

BENCHMARK(BM_Gustavson_Nemeth)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_Nemeth)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_Nemeth)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_Lhr71c)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_Lhr71c)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_Lhr71c)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_C71)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_C71)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_C71)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_preferentialAttachment)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_preferentialAttachment)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_preferentialAttachment)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_consph)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_consph)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_consph)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_ASIC_680ks)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_ASIC_680ks)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_ASIC_680ks)->Repetitions(repeats)->ReportAggregatesOnly(true);

// Expensive benchmarks...
BENCHMARK(BM_Gustavson_rgg_n_2_22_s0)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_rgg_n_2_22_s0)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_rgg_n_2_22_s0)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_rajat31)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_rajat31)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_rajat31)->Repetitions(repeats)->ReportAggregatesOnly(true);

BENCHMARK(BM_Gustavson_M6)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Parallel_M6)->Repetitions(repeats)->ReportAggregatesOnly(true);
BENCHMARK(BM_Eigen_M6)->Repetitions(repeats)->ReportAggregatesOnly(true);

// -+--+--+--+--+--+--+--+--+--+--+--+- Scaling -+--+--+--+--+--+--+--+--+--+--+--+- //
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 1)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 2)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 4)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 8)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 16)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048, 32)->Arg(85)->Arg(90)->Arg(95)->Arg(99);

// -+--+--+--+--+--+--+--+--+--+--+--+- Dtype == int -+--+--+--+--+--+--+--+--+--+--+--+- //
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Gustavson_random, int, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);

BENCHMARK_TEMPLATE(BM_Parallel_random, int, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Parallel_random, int, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);

BENCHMARK_TEMPLATE(BM_Eigen_random, int, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Eigen_random, int, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Eigen_random, int, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Eigen_random, int, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Eigen_random, int, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
BENCHMARK_TEMPLATE(BM_Eigen_random, int, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);

// -+--+--+--+--+--+--+--+--+--+--+--+- Dtype = double -+--+--+--+--+--+--+--+--+--+--+--+- //
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Gustavson_random, double, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// 
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Parallel_random, double, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// 
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 64)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 128)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 256)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 512)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 1024)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
// BENCHMARK_TEMPLATE(BM_Eigen_random, double, 2048)->Arg(85)->Arg(90)->Arg(95)->Arg(99);
