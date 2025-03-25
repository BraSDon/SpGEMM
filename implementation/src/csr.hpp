#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "common.h"

template <class T> class CSRMatrix {
private:
  size_t rows_, cols_;
  std::vector<T> values_;
  std::vector<size_t> col_idx_;
  // NOTE: since its size is known, we could use std::array with template
  // parameter. However, this would make reading from file more complex...
  std::vector<size_t> row_ptr_;

public:
  // -+--+--+--+--+--+--+--+--+--+--+- Constructors -+--+--+--+--+--+--+--+--+--+--+--+- //
  CSRMatrix (const CSRMatrix&) = delete;
  CSRMatrix(CSRMatrix &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_),
        values_(std::move(other.values_)), col_idx_(std::move(other.col_idx_)),
        row_ptr_(std::move(other.row_ptr_)) {}

  // Default constructor
  CSRMatrix() noexcept : rows_(0), cols_(0) { row_ptr_.push_back(0); }

  // TODO: add heuristic for reservering memory (typically done via density estimation)
  // but that is out of scope.
  CSRMatrix(size_t rows, size_t cols) noexcept : rows_(rows), cols_(cols) {
    row_ptr_.resize(rows + 1, 0);
  }

  // Known nnz constructor
  CSRMatrix(size_t rows, size_t cols, size_t nnz)
      : rows_(rows), cols_(cols), values_(nnz), col_idx_(nnz) {
    row_ptr_.resize(rows + 1, 0);
  }

  // From dense matrix
  explicit CSRMatrix(const DenseMatrix<T> &matrix) noexcept {
    // Handle edge case: empty matrix
    if (matrix.empty() || matrix[0].empty()) {
      rows_ = cols_ = 0;
      row_ptr_.push_back(0);
      return;
    }

    rows_ = matrix.size();
    cols_ = matrix[0].size();
    row_ptr_.resize(rows_ + 1, 0);

    // Build CSR structure
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        if (matrix[row][col] != static_cast<T>(0)) {
          values_.push_back(matrix[row][col]);
          col_idx_.push_back(col);
        }
      }
      row_ptr_[row + 1] = values_.size();
    }
  }

  // From triple (i, j, value) constructor
  explicit CSRMatrix(const TripleMatrix<T> &triples, size_t rows,
                     size_t cols) noexcept {
    this->rows_ = rows;
    this->cols_ = cols;
    values_.reserve(triples.size());
    col_idx_.reserve(triples.size());
    row_ptr_.resize(rows + 1, 0);

    // Make a copy of the triples and sort them
    auto sorted_triples = triples;
    std::sort(sorted_triples.begin(), sorted_triples.end(),
              [](const auto &a, const auto &b) {
                // Sort primarily by row (i), secondarily by column (j)
                return std::tie(std::get<0>(a), std::get<1>(a)) <
                       std::tie(std::get<0>(b), std::get<1>(b));
              });

    // Build CSR structure from sorted triples
    for (const auto &[i, j, value] : sorted_triples) {
      values_.push_back(value);
      col_idx_.push_back(j);
      ++row_ptr_[i + 1];
    }

    // Accumulate row_ptr_ to represent row offsets
    for (size_t row = 0; row < rows; ++row) {
      row_ptr_[row + 1] += row_ptr_[row];
    }
  }
  // -+--+--+--+--+--+--+--+--+--+--+--+- Conversion functions -+--+--+--+--+--+--+--+--+--+--+--+- //
  DenseMatrix<T> to_dense() const {
    DenseMatrix<T> dense = DenseMatrix(rows_, std::vector<T>(cols_, 0));
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t idx = row_ptr_[row]; idx < row_ptr_[row + 1]; ++idx) {
        dense[row][col_idx_[idx]] = values_[idx];
      }
    }
    return dense;
  }

  TripleMatrix<T> to_triple() const {
    TripleMatrix<T> triples;
    triples.reserve(values_.size());
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t idx = row_ptr_[row]; idx < row_ptr_[row + 1]; ++idx) {
        triples.emplace_back(row, col_idx_[idx], values_[idx]);
      }
    }
    return triples;
  }

  Eigen::SparseMatrix<T> to_eigen() const {
    Eigen::SparseMatrix<T> matrix(static_cast<long int>(rows_),
                                  static_cast<long int>(cols_));
    TripleMatrix<T> triples = to_triple();
    auto triplets = to_eigen_triplets(triples);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
  }

  void to_mm_file(const std::string &path) const {
    TripleMatrix<T> triples = to_triple();
    CSRMatrix<T>::to_mm_file(path, triples, rows_, cols_);
  }

  // -+--+--+--+--+--+--+--+--+--+--+- Static functions -+--+--+--+--+--+--+--+--+--+--+--+- //
  static void to_mm_file(const std::string &path,
                         const TripleMatrix<T> &triples, size_t rows,
                         size_t cols) {
    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + path);
    }

    bool is_symmetric = true;
    std::map<std::pair<size_t, size_t>, T> map;
    for (const auto &[row, col, value] : triples) {
      map[{row, col}] = value;
    }
    for (const auto &[key, val] : map) {
      auto [row, col] = key;
      auto it = map.find({col, row});
      if (it == map.end() || it->second != val) {
        is_symmetric = false;
        break;
      }
    }
    // remove elements above diagonal if symmetric matrix
    TripleMatrix<T> symmetric_triples;
    if (is_symmetric) {
      for (const auto &[row, col, value] : triples) {
        if (row <= col) {
          symmetric_triples.emplace_back(row, col, value);
        }
      }
    }

    file << "%%MatrixMarket matrix coordinate "
         << (std::is_integral<T>::value ? "integer" : "real") << " "
         << (is_symmetric ? "symmetric" : "general") << "\n";

    if (is_symmetric) {
      file << rows << " " << cols << " " << symmetric_triples.size() << "\n";
      for (const auto &[row, col, value] : symmetric_triples) {
        std::cout << row + 1 << " " << col + 1 << " " << value << std::endl;
        file << row + 1 << " " << col + 1 << " " << value
             << "\n"; // 1-based indexing for MM format
      }
    } else {
      file << rows << " " << cols << " " << triples.size() << "\n";
      for (const auto &[row, col, value] : triples) {
        file << row + 1 << " " << col + 1 << " " << value
             << "\n"; // 1-based indexing for MM format
      }
    }

    file.close();
  }

  static CSRMatrix<T> from_mm_file(const std::string &path) {
    class MMFormat {
    public:
        enum class ObjectType { matrix, vector };
        enum class Format { coordinate, array };
        enum class Field { real, ddouble, complex, integer, pattern };
        enum class Symmetry { general, symmetric, skew_symmetric, hermitian };

        MMFormat(std::ifstream& infile) {
            std::string line;
            std::getline(infile, line);
            std::istringstream header_stream(line);
            std::string token, object_str, format_str, dtype_str, symmetry_str;
            header_stream >> token >> object_str >> format_str >> dtype_str >> symmetry_str;

            this->object = parse_object_type(object_str);
            this->format = parse_format(format_str);
            this->field = parse_field(dtype_str);
            this->symmetry = parse_symmetry(symmetry_str);

            validate_dtype();
        }

        void validate_dtype() const {
            if ((field == Field::integer || field == Field::pattern) && !std::is_integral<T>::value) {
                throw std::runtime_error("Incompatible data type.");
            }
            if ((field == Field::real || field == Field::ddouble) && !std::is_floating_point<T>::value) {
                throw std::runtime_error("Incompatible data type.");
            }
        }

        bool is_pattern() const {
            return field == Field::pattern;
        }

        bool is_symmetric() const {
            return symmetry == Symmetry::symmetric;
        }

    private:
        ObjectType object;
        Format format;
        Field field;
        Symmetry symmetry;

        static ObjectType parse_object_type(const std::string& str) {
            if (str == "matrix") return ObjectType::matrix;
            if (str == "vector") throw std::runtime_error("Vector object not supported.");
            throw std::runtime_error("Invalid object type: " + str);
        }

        static Format parse_format(const std::string& str) {
            if (str == "coordinate") return Format::coordinate;
            if (str == "array") throw std::runtime_error("Array format not supported.");
            throw std::runtime_error("Invalid format type: " + str);
        }

        static Field parse_field(const std::string& str) {
            if (str == "real") return Field::real;
            if (str == "double") return Field::ddouble;
            if (str == "complex") throw std::runtime_error("Complex field not supported.");
            if (str == "integer") return Field::integer;
            if (str == "pattern") return Field::pattern;
            throw std::runtime_error("Invalid field type: " + str);
        }

        static Symmetry parse_symmetry(const std::string& str) {
            if (str == "general") return Symmetry::general;
            if (str == "symmetric") return Symmetry::symmetric;
            if (str == "skew-symmetric") throw std::runtime_error("Skew-symmetric symmetry not supported.");
            if (str == "hermitian") throw std::runtime_error("Hermitian symmetry not supported.");
            throw std::runtime_error("Invalid symmetry type: " + str);
        }
    };

    std::ifstream infile(path);
    if (!infile) {
      throw std::runtime_error("Unable to open file: " + path);
    }

    MMFormat mm_format(infile);

    std::string line;

    // Skip comments
    while (std::getline(infile, line) && line[0] == '%') {
      continue;
    }

    size_t rows, cols, nnz;
    std::istringstream dimension_stream(line);
    if (!(dimension_stream >> rows >> cols >> nnz)) {
      throw std::runtime_error("Error reading matrix dimensions from file: " +
                               path);
    }

    size_t unique_entries = 0;
    TripleMatrix<T> matrix;
    matrix.reserve(nnz);

    while (std::getline(infile, line)) {
      std::istringstream entry_stream(line);
      size_t row, col;
      T value;

      if (mm_format.is_pattern()) {
        if (!(entry_stream >> row >> col)) {
          throw std::runtime_error("Error reading matrix entry from file: " +
                                   path);
        }
        value = static_cast<T>(1);
      } else {
        if (!(entry_stream >> row >> col >> value)) {
          throw std::runtime_error("Error reading matrix entry from file: " +
                                   path);
        }
      }
      matrix.emplace_back(row - 1, col - 1, value);
      unique_entries++;
      if (mm_format.is_symmetric() && row != col) {
        matrix.emplace_back(col - 1, row - 1, value);
      }
    }
    if (unique_entries != nnz) {
      throw std::runtime_error("Expected " + std::to_string(nnz) +
                               " entries, but read " +
                               std::to_string(matrix.size()));
    }

    return CSRMatrix<T>(matrix, rows, cols);
  }
  // -+--+--+--+--+--+--+--+--+--+--+- Operators -+--+--+--+--+--+--+--+--+--+--+--+- // 
  CSRMatrix& operator= (const CSRMatrix&) = delete;
  CSRMatrix<T> &operator=(CSRMatrix<T> &&other) noexcept {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      values_ = std::move(other.values_);
      col_idx_ = std::move(other.col_idx_);
      row_ptr_ = std::move(other.row_ptr_);
    }
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const CSRMatrix<T> &matrix) {
    os << "CSRMatrix (" << matrix.get_rows() << " x " << matrix.get_cols()
       << ")\n";
    os << "Values: ";
    for (const auto &val : matrix.values()) {
      os << val << " ";
    }
    os << "\nRow Indices: ";
    for (const auto &row : matrix.row_ptr()) {
      os << row << " ";
    }
    os << "\nColumn Indices: ";
    for (const auto &col : matrix.col_idx()) {
      os << col << " ";
    }
    os << "\n";
    return os;
  }

  bool operator==(const CSRMatrix<T> &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      return false;
    }
    if (row_ptr_ != other.row_ptr_ || col_idx_ != other.col_idx_) {
      return false;
    }

    if constexpr (std::is_floating_point<T>::value) {
      const double epsilon = 1e-9;
      for (size_t i = 0; i < values_.size(); ++i) {
        if (std::abs(values_[i] - other.values_[i]) > epsilon) {
          return false;
        }
      }
    } else {
      if (values_ != other.values_) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const CSRMatrix<T> &other) const { return !(*this == other); }

  // -+--+--+--+--+--+--+--+--+--+--+- Getters -+--+--+--+--+--+--+--+--+--+--+--+- //
  size_t get_rows() const { return rows_; }
  size_t get_cols() const { return cols_; }
  size_t nnz() const { return values_.size(); }
  const std::vector<T> &values() const { return values_; }
  const std::vector<size_t> &col_idx() const { return col_idx_; }
  const std::vector<size_t> &row_ptr() const { return row_ptr_; }

  std::vector<T>& mut_values() { return values_; }
  std::vector<size_t>& mut_col_idx() { return col_idx_; }
  std::vector<size_t>& mut_row_ptr() { return row_ptr_; }

  // NOTE: Needed for gustavson algorithm, might be refactorable
  void set_row_ptr_entry(size_t idx, size_t value) { row_ptr_[idx] = value; }
  void append_col_idx(size_t value) { col_idx_.push_back(value); }
  void append_value(T value) { values_.push_back(value); }

  void print_dense() const {
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        size_t idx = row_ptr_[row];
        if (idx < row_ptr_[row + 1] && col_idx_[idx] == col) {
          std::cout << values_[idx] << " ";
          ++idx;
        } else {
          std::cout << "0 ";
        }
      }
      std::cout << std::endl;
    }
  }

  void print_sparse() const {
    std::cout << "Values: ";
    for (const auto &value : values_) {
      std::cout << value << " ";
    }
    std::cout << std::endl;

    std::cout << "Col_idx: ";
    for (const auto &idx : col_idx_) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;

    std::cout << "Row_ptr: ";
    for (const auto &ptr : row_ptr_) {
      std::cout << ptr << " ";
    }
    std::cout << std::endl;
  }
};
