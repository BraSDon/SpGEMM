#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <tuple>

template <class T>
using DenseMatrix = std::vector<std::vector<T>>;

template <class T>
using TripleMatrix = std::vector<std::tuple<size_t, size_t, T>>;

template <class T>
std::vector<Eigen::Triplet<T>> to_eigen_triplets(const TripleMatrix<T>& triples) {
    std::vector<Eigen::Triplet<T>> eigen_triplets;
    eigen_triplets.reserve(triples.size());

    for (const auto& [row, col, value] : triples) {
        eigen_triplets.emplace_back(row, col, value);
    }

    return eigen_triplets;
}

// Custom hash function for std::pair<size_t, size_t>
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const noexcept {
    std::size_t h1 = std::hash<T1>{}(pair.first);
    std::size_t h2 = std::hash<T2>{}(pair.second);
    return h1 ^ (h2 << 1); // Combine the two hashes
  }
};
