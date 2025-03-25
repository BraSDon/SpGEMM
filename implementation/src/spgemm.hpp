#pragma once

#include "common.h"
#include "csr.hpp"
#include <cstdint>
#include <omp.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

// -+--+--+--+--+--+--+--+--+--+--+--+- Gustavson Algorithm -+--+--+--+--+--+--+--+--+--+--+--+- //
template <class T, size_t N, size_t NUM_THREADS = 4>
CSRMatrix<T> gustavson_parallel(const CSRMatrix<T> &A, const CSRMatrix<T> &B) {
  if (A.get_cols() != B.get_rows()) {
    throw std::invalid_argument("Matrix dimensions do not match");
  }
  if (N != B.get_cols()) {
    throw std::invalid_argument(
        "Template parameter n must match the number of columns of B");
  }

  std::array<CSRMatrix<T>, NUM_THREADS> local_results;

  // Optimized access to CSRMatrix data
  const auto &IA = A.row_ptr();
  const auto &JA = A.col_idx();
  const auto &A_val = A.values();

  const auto &IB = B.row_ptr();
  const auto &JB = B.col_idx();
  const auto &B_val = B.values();

  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
  {
    size_t tid = static_cast<size_t>(omp_get_thread_num());
    std::vector<size_t> xb(N, SIZE_MAX);
    std::vector<T> x(N);
    CSRMatrix<T> C_local = CSRMatrix<T>(A.get_rows(), B.get_cols());
    auto &IC = C_local.row_ptr();
    auto &JC = C_local.col_idx();
    size_t ip = 0;
    size_t nnz_row = 0;

    // NOTE: 2048 was optimized for the large florida matrices, but for smaller matrices
    // this is obviously too large. To get a compile time constant, we use N / NUM_THREADS
    // as a heuristic to determine the chunk size. Optimal would be static scheduling, however
    // that would make the code quite verbose.
    constexpr size_t CHUNK_SIZE = (N / NUM_THREADS < 2048) ? (N / NUM_THREADS) : 2048;

    #pragma omp for schedule(monotonic: dynamic, CHUNK_SIZE) nowait
    for (size_t i = 0; i < IA.size() - 1; ++i) {
      nnz_row = 0;
      size_t start_row_A = IA[i];
      size_t end_row_A = IA[i + 1];
      for (size_t jp = start_row_A; jp < end_row_A; ++jp) {
        size_t j = JA[jp];
        size_t start_row_B = IB[j];
        size_t end_row_B = IB[j + 1];
        T A_val_jp = A_val[jp];
        for (size_t kp = start_row_B; kp < end_row_B; ++kp) {
          size_t k = JB[kp];
          T B_val_kp = B_val[kp];
          if (xb[k] != i) {
            C_local.append_col_idx(k);
            ++nnz_row;
            xb[k] = i;
            x[k] = A_val_jp * B_val_kp;
          } else {
            x[k] += A_val_jp * B_val_kp;
          }
        }
      }
      // NOTE: instead of setting i-th entry as in sequential version, we set the i+1-th entry
      // because this thread might not be responsible for the next row.
      ip += nnz_row;
      C_local.set_row_ptr_entry(i + 1, ip);

      for (size_t vp = IC[i + 1] - nnz_row; vp < ip; ++vp) {
        size_t v = JC[vp];
        C_local.append_value(x[v]);
      }
    }

    // NOTE: because each thread might not have a continous range of rows, we might have holes
    // in the sense of zeros in the row_ptr, even tho it should be a prefix sum.
    // To fix this we simply fill up the holes with the last seen value on the left to uphold the
    // prefix sum property.
    size_t left = 0;
    auto& row_ptr = C_local.mut_row_ptr();
    for (auto& p : row_ptr) {
      if (p == 0) {
        p = left;
      } else {
        left = p;
      }
    }

    // move local result to shared vector
    local_results[tid] = std::move(C_local);
  }

  // -+--+--+--+--+--+--+--+--+--+--+- Merge local C's -+--+--+--+--+--+--+--+--+--+--+--+- //
  // calculate global nnz, to preallocate memory
  size_t nnz = 0;
  for (const auto& C_local : local_results) {
    nnz += C_local.nnz();
  }
  CSRMatrix<T> C = CSRMatrix<T>(A.get_rows(), B.get_cols(), nnz);

  auto& values = C.mut_values();
  auto& col_idx = C.mut_col_idx();
  auto& row_ptr = C.mut_row_ptr();

  // compute global C's row_ptr by adding up local row_ptrs
  // NOTE: loop order swapped to avoid repeatedly creating parallel region
  // NOTE: manual loop unrolling did not yield significant performance improvements
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < row_ptr.size(); ++i) {
    for (const auto& C_local : local_results) {
      row_ptr[i] += C_local.row_ptr()[i];
    }
  }

  struct Batch {
      long int local_start;   // local_row_ptr[start_row]
      long int local_end;     // local_row_ptr[end_row]
      long int global_start;  // row_ptr[start_row]
  };

  // copy local C's to global C, row by row.
  #pragma omp parallel for schedule(static)
  for (const auto& C_local : local_results) {
    const auto& local_values = C_local.values();
    const auto& local_col_idx = C_local.col_idx();
    const auto& local_row_ptr = C_local.row_ptr();

    std::vector<Batch> batches;
    const size_t num_rows = A.get_rows();
    size_t row = 0;
    while (row < num_rows) {
        // skip empty rows
        if (local_row_ptr[row] == local_row_ptr[row + 1]) {
            ++row;
            continue;
        }

        size_t batch_start = row;
        size_t batch_end = row + 1;
        // try to extend the batch
        while (batch_end < num_rows) {
            if (local_row_ptr[batch_end] == local_row_ptr[batch_end + 1]) break;
            ++batch_end;
        }
        Batch b;
        b.local_start  = static_cast<long int>(local_row_ptr[batch_start]);
        b.local_end    = static_cast<long int>(local_row_ptr[batch_end]);
        b.global_start = static_cast<long int>(row_ptr[batch_start]);
        batches.push_back(b);

        row = batch_end;
    }

    for (const auto& b : batches) {
      const auto batch_nnz = b.local_end - b.local_start;
      std::copy_n(local_values.begin() + b.local_start, batch_nnz, values.begin() + b.global_start);
      std::copy_n(local_col_idx.begin() + b.local_start, batch_nnz, col_idx.begin() + b.global_start);
    }
  }

  assert(C.row_ptr().size() == IA.size());
  return C;
}

template <class T, size_t N>
CSRMatrix<T> gustavson(const CSRMatrix<T> &A, const CSRMatrix<T> &B) {
  if (A.get_cols() != B.get_rows()) {
    throw std::invalid_argument("Matrix dimensions do not match");
  }
  if (N != B.get_cols()) {
    throw std::invalid_argument(
        "Template parameter n must match the number of columns of B");
  }

  CSRMatrix<T> C = CSRMatrix<T>(A.get_rows(), B.get_cols());
  size_t ip = 0;
  // NOTE: xb/x could be arrays, but they would not fit in stack memory for large N.
  // Also arrays are not significantly faster according to my benchmarks.
  std::vector<size_t> xb(N, SIZE_MAX);
  std::vector<T> x(N);

  // NOTE: Defining const variables outside of the loops improves performance.
  const auto &IA = A.row_ptr();
  const auto &JA = A.col_idx();
  const auto &A_val = A.values();

  const auto &IB = B.row_ptr();
  const auto &JB = B.col_idx();
  const auto &B_val = B.values();

  for (size_t i = 0; i < IA.size() - 1; ++i) {
    C.set_row_ptr_entry(i, ip);
    size_t start_row_A = IA[i];
    size_t end_row_A = IA[i + 1];
    for (size_t jp = start_row_A; jp < end_row_A; ++jp) {
      size_t j = JA[jp];
      size_t start_row_B = IB[j];
      size_t end_row_B = IB[j + 1];
      T A_val_jp = A_val[jp];
      for (size_t kp = start_row_B; kp < end_row_B; ++kp) {
        size_t k = JB[kp];
        T B_val_kp = B_val[kp];
        if (xb[k] != i) {
          C.append_col_idx(k);
          ++ip;
          xb[k] = i;
          x[k] = A_val_jp * B_val_kp;
        } else {
          // NOTE: this allows for a fused multiply-add (FMA) instruction
          // If it is actually faster than computing the product before the if-statement
          // depends on how often the branch is taken.
          x[k] += A_val_jp * B_val_kp;
        }
      }
    }
    const auto &IC = C.row_ptr();
    const auto &JC = C.col_idx();
    for (size_t vp = IC[i]; vp < ip; ++vp) {
      size_t v = JC[vp];
      C.append_value(x[v]);
    }
  }
  assert(C.row_ptr().size() == IA.size());
  C.set_row_ptr_entry(IA.size() - 1, ip);
  return C;
}

// -+--+--+--+--+--+--+--+--+--+--+--+- Join Aggregate Algorithm -+--+--+--+--+--+--+--+--+--+--+--+- //
template <class Dtype> struct JoinAggregate {
  TripleMatrix<Dtype> operator()(const CSRMatrix<Dtype> &A,
                                 const CSRMatrix<Dtype> &B) const {
    return join_aggregate(A, B);
  }
};

template <class T>
TripleMatrix<T> join_aggregate(const CSRMatrix<T> &A, const CSRMatrix<T> &B) {
  TripleMatrix<T> result;
  const size_t n = A.get_rows();

  const auto &A_values = A.values();
  const auto &A_col_idx = A.col_idx();
  const auto &A_row_ptr = A.row_ptr();

  const auto &B_values = B.values();
  const auto &B_col_idx = B.col_idx();
  const auto &B_row_ptr = B.row_ptr();

  for (size_t i = 0; i < n; ++i) {
    size_t A_start_row = A_row_ptr[i];
    size_t A_end_row = A_row_ptr[i + 1];
    for (size_t j = A_start_row; j < A_end_row; ++j) {
      size_t B_start_row = B_row_ptr[A_col_idx[j]];
      size_t B_end_row = B_row_ptr[A_col_idx[j] + 1];
      for (size_t k = B_start_row; k < B_end_row; ++k) {
        result.emplace_back(i, B_col_idx[k], A_values[j] * B_values[k]);
      }
    }
  }

  // Aggregate phase: Combine duplicates
  std::unordered_map<std::pair<size_t, size_t>, T, PairHash> aggregate_map;
  for (const auto &[i, j, value] : result) {
    aggregate_map[{i, j}] += value;
  }

  // Convert map to final result vector
  TripleMatrix<T> aggregated_result;
  for (const auto &[key, value] : aggregate_map) {
    aggregated_result.emplace_back(key.first, key.second, value);
  }

  return aggregated_result;
}
