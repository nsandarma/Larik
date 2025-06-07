#ifndef _LARIK_HPP
#define _LARIK_HPP

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vecLib/cblas.h>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#error "BLAS operations via Accelerate are only supported on macOS"
#endif

template <typename T> class Tensor {
private:
  std::vector<T> owned_buffer_;
  T *buffer_;
  size_t size_;
  std::vector<size_t> shape_;
  bool owns_buffer_; // Flag to track buffer ownership
                     //
  // Private helper to validate shape and size consistency
  void validate_shape() const {
    size_t computed_size = std::accumulate(
        shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    if (computed_size != size_) {
      throw std::invalid_argument("Shape dimensions do not match buffer size");
    }
  }

  // Helper to compute flat index from multi-dimensional indices
  size_t compute_index(const std::vector<size_t> &indices) const {
    if (indices.size() != shape_.size()) {
      throw std::invalid_argument("Index dimensions do not match tensor shape");
    }
    size_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= shape_[i]) {
        throw std::out_of_range("Index out of bounds");
      }
      idx = idx * shape_[i] + indices[i];
    }
    return idx;
  }

  std::string get_dtype() const {
    if constexpr (std::is_same<T, float>::value)
      return "float32";
    if constexpr (std::is_same<T, double>::value)
      return "float64";
    if constexpr (std::is_same<T, int>::value)
      return "int32";
    if constexpr (std::is_same<T, long>::value)
      return "int64";
    return typeid(T).name();
  }

public:
  // helpers
  bool is_valid() const { return buffer_ != nullptr && size_ > 0; }
  bool is_owned_and_valid() const {
    return owns_buffer_ && buffer_ != nullptr && size_ > 0;
  }
  bool empty() const { return size_ == 0 || buffer_ == nullptr; }

  // Constructor dengan owned buffer (move)
  Tensor(const std::vector<size_t> &shape, std::vector<T> &&buffer)
      : owned_buffer_(std::move(buffer)), buffer_(owned_buffer_.data()),
        size_(owned_buffer_.size()), shape_(shape), owns_buffer_(true) {
    validate_shape();

#ifdef DBG
    std::cout << "owned buffer" << std::endl;
#endif
  }

  // Constructor dengan external buffer (tidak memiliki)
  Tensor(const std::vector<size_t> &shape, T *external_buffer, size_t size)
      : buffer_(external_buffer), size_(size), shape_(shape),
        owns_buffer_(false) {
    if (!external_buffer) {
      throw std::invalid_argument("External buffer cannot be null");
    }
    validate_shape();

#ifdef DBG
    std::cout << "external buffer" << std::endl;
#endif
  }

  // Copy Constructor
  Tensor(const Tensor &other)
      : owned_buffer_(other.owns_buffer_ ? other.owned_buffer_
                                         : std::vector<T>()),
        buffer_(other.owns_buffer_ ? owned_buffer_.data() : other.buffer_),
        size_(other.size_), shape_(other.shape_),
        owns_buffer_(other.owns_buffer_) {

#ifdef DBG
    std::cout << "copy constructor" << std::endl;
#endif
  }

  // Move constructor
  Tensor(Tensor &&other) noexcept
      : owned_buffer_(std::move(other.owned_buffer_)), buffer_(other.buffer_),
        size_(other.size_), shape_(std::move(other.shape_)),
        owns_buffer_(other.owns_buffer_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.owns_buffer_ = false;

#ifdef DBG
    std::cout << "move constructor" << std::endl;
#endif
  }

  // Copy assignment
  Tensor &operator=(const Tensor &other) {
    if (this != &other) {
      owns_buffer_ = other.owns_buffer_; // harus set ini dulu
      if (owns_buffer_) {
        owned_buffer_ = other.owned_buffer_;
        buffer_ = owned_buffer_.data();
      } else {
        owned_buffer_.clear();
        buffer_ = other.buffer_;
      }
      size_ = other.size_;
      shape_ = other.shape_;
    }
    return *this;
  }

  // Move assignment
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      owned_buffer_ = std::move(other.owned_buffer_);
      buffer_ = other.buffer_;
      size_ = other.size_;
      shape_ = std::move(other.shape_);
      owns_buffer_ = other.owns_buffer_;
      other.buffer_ = nullptr;
      other.size_ = 0;
      other.owns_buffer_ = false;
    }
    return *this;
  }

  // Destructor
  ~Tensor() {
    if (!owns_buffer_) {
      buffer_ = nullptr; // Prevent accidental deletion of non-owned buffer
    }
  }

  // Accessors
  T *data() noexcept { return buffer_; }
  const T *data() const noexcept { return buffer_; }
  size_t size() const noexcept { return size_; }
  size_t dim() const noexcept { return shape_.size(); }
  const std::vector<size_t> &shape() const noexcept { return shape_; }

  // Array-like access
  T &operator[](size_t i) {
    if (i >= size_) {
      throw std::out_of_range("Index out of range");
    }
    return buffer_[i];
  }

  const T &operator[](size_t i) const {
    if (i >= size_) {
      throw std::out_of_range("Index out of range");
    }
    return buffer_[i];
  }

  // Multi-dimensional access
  T &operator()(const std::vector<size_t> &indices) {
    size_t idx = compute_index(indices);
    return buffer_[idx];
  }

  const T &operator()(const std::vector<size_t> &indices) const {
    size_t idx = compute_index(indices);
    return buffer_[idx];
  }

  template <typename U> Tensor<U> cast() const {
    std::vector<U> new_buffer(size_);
    std::transform(buffer_, buffer_ + size_, new_buffer.begin(),
                   [](const T &val) { return static_cast<U>(val); });
    return Tensor<U>(shape_, std::move(new_buffer));
  }

  void print(bool endl = false) const {
    // if (!is_valid()) {
    //   std::cout << "Empty tensor\n";
    //   return;
    // }
    if (shape_.size() == 1) {
      // 1D tensor
      std::cout << "[ ";
      for (size_t i = 0; i < size_; ++i) {
        std::cout << buffer_[i] << " ";
      }
      std::cout << "]\n";
    } else if (shape_.size() == 2) {
      // 2D tensor
      size_t rows = shape_[0];
      size_t cols = shape_[1];
      if (rows * cols != size_) {
        std::cerr << "Shape and size mismatch in 2D print\n";
        return;
      }
      for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
          std::cout << buffer_[i * cols + j] << " ";
        }
        std::cout << "]\n";
      }
    } else {
      std::cout << "Tensor with " << shape_.size()
                << " dimensions (printing not supported yet)\n";
    }
    if (endl) {
      std::cout << "\n";
    }
  }

  void info() const {
    std::cout << "(";
    for (const auto &elem : shape_) {
      std::cout << elem << " ";
    }
    std::cout << ")" << std::endl;
    std::cout << "size : " << size() << " | dims : " << dim() << std::endl;
    std::cout << "Owns Buffer :" << (owns_buffer_ ? "Yes" : "No") << std::endl;
  }

  Tensor<T> matmul_int(const Tensor<T> &other) const {
    Tensor<float> x = cast<float>();
    Tensor<float> y = other.cast<float>();
    Tensor<float> z = x.matmul(y);
    return z.cast<T>();
  }

  Tensor<T> matmul(const Tensor<T> &other) const {
    if (dim() != 2 || other.dim() != 2) {
      throw std::invalid_argument(
          "Tensor Multiplication requires 2D tensors 1");
    }
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = other.shape_[1];
    if (k != other.shape_[0]) {
      throw std::invalid_argument(
          "Tensor dimensions are not compatible for multiplications");
    }

    std::vector<size_t> result_shape = {m, n};
    Tensor<T> result(result_shape, std::vector<T>(m * n, T(0)));

    if constexpr (std::is_same<T, float>::value) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                  1.0f,                                            // alpha
                  buffer_, static_cast<int>(k),                    // A, lda
                  other.buffer_, static_cast<int>(n),              // B, ldb
                  0.0f,                                            // beta
                  result.owned_buffer_.data(), static_cast<int>(n) // C, ldc
      );
    } else if constexpr (std::is_same<T, double>::value) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                  1.0,                                             // alpha
                  buffer_, static_cast<int>(k),                    // A, lda
                  other.buffer_, static_cast<int>(n),              // B, ldb
                  0.0,                                             // beta
                  result.owned_buffer_.data(), static_cast<int>(n) // C, ldc
      );
    } else if constexpr (std::is_integral<T>::value) {
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
          T sum = T(0);
          for (size_t l = 0; l < k; ++l) {
            sum += buffer_[i * k + l] * other.buffer_[l * n + j];
          }
          result.buffer_[i * n + j] = sum;
        }
      }
    } else {
      throw std::invalid_argument("Tensor multiplications only supports "
                                  "float,double, or integral types.");
    }
    return result;
  }
  Tensor<T> multiply(const Tensor<T> &other) const {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("shape of tensor is not same!");
    }
    size_t m = shape_[0];
    size_t n = shape_[1];

    Tensor<T> result(shape_, std::vector<T>(m * n, T(0)));

    for (size_t i = 0; i < size(); ++i) {
      result[i] = buffer_[i] * other[i];
    }
    return result;
  }
  Tensor<T> sub(const Tensor<T> &other) const {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("shape of tensor is not same!");
    }
    size_t m = shape_[0];
    size_t n = shape_[1];
    Tensor<T> result(shape_, std::vector<T>(m * n, T(0)));
    for (size_t i = 0; i < size(); i++) {
      result[i] = buffer_[i] - other[i];
    }
    return result;
  }
  template <typename U> Tensor<T> sub(U value) const {
    size_t m = shape_[0];
    size_t n = shape_[1];
    Tensor<T> result(shape_, std::vector<T>(m * n, T(0)));
    for (size_t i = 0; i < size(); i++) {
      result[i] = buffer_[i] - value;
    }
    return result;
  }
  Tensor<T> add(const Tensor<T> &other) const {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shape of tensor is not the same!");
    }

    Tensor<T> result(shape_, std::vector<T>(size_, T(0)));

    // Copy buffer_ to result first
    std::copy(buffer_, buffer_ + size_, result.owned_buffer_.data());

    // BLAS AXPY: result = buffer_ + other
    if constexpr (std::is_same<T, float>::value) {
      cblas_saxpy(static_cast<int>(size_), 1.0f, other.data(), 1,
                  result.owned_buffer_.data(), 1);
    } else if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(static_cast<int>(size_), 1.0, other.data(), 1,
                  result.owned_buffer_.data(), 1);
    } else {
      for (size_t i = 0; i < size_; ++i) {
        result[i] = buffer_[i] + other[i];
      }
    }

    return result;
  }

  template <typename U> Tensor<T> add(U value) const {
    Tensor<T> result(shape_, std::vector<T>(size_, T(0)));
    for (size_t i = 0; i < size_; i++) {
      result[i] = buffer_[i] + value;
    }
    return result;
  }

  static Tensor<T> arange(const T start, const T stop, const T step) {
    if (step == T(0)) {
      throw std::invalid_argument("Step must not be zero");
    }

    std::vector<T> data;
    if (step > 0) {
      for (T val = start; val < stop; val += step) {
        data.push_back(val);
      }
    } else {
      for (T val = start; val > stop; val += step) {
        data.push_back(val);
      }
    }

    std::vector<size_t> shape = {data.size()};
    return Tensor<T>(shape, std::move(data));
  }

  static Tensor<T> zeros(const std::vector<size_t> &shape) {
    size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                  std::multiplies<size_t>());
    std::vector<T> buff = std::vector<T>(size, T(0));
    return Tensor<T>(shape, std::move<>);
  }

  static Tensor<T> rand(const std::vector<size_t> &shape,
                        unsigned int seed = 0) {
    size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                  std::multiplies<size_t>());
    std::vector<T> buffer(size);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
      buffer[i] = dist(rng);
    }
    return Tensor<T>(shape, std::move(buffer));
  }
  static Tensor<T> randn(const std::vector<size_t> &shape, T mean = T(0),
                         T stddev = T(1), unsigned int seed = 0) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value,
                  "randn is only supported for float or double types");
    if (shape.empty()) {
      throw std::invalid_argument("Shape cannot be empty");
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                  std::multiplies<size_t>());
    std::vector<T> buffer(size);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<T> dist(mean, stddev);
    for (size_t i = 0; i < size; ++i) {
      buffer[i] = dist(rng);
    }
    return Tensor<T>(shape, std::move(buffer));
  }
  static Tensor<T> randint(const std::vector<size_t> &shape, T low, T high,
                           unsigned int seed = 0) {
    static_assert(std::is_integral<T>::value,
                  "randint is only supported integral types");
    if (shape.empty()) {
      throw std::invalid_argument("Shape cannot be empty !");
    }
    if (high <= low) {
      throw std::invalid_argument("High must be greater than low");
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                  std::multiplies<size_t>());
    std::vector<T> buffer(size);
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<T> dist(low, high - 1);
    for (size_t i = 0; i < size; ++i) {
      buffer[i] = dist(rng);
    }
    return Tensor<T>(shape, std::move(buffer));
  }

  // Export
  void export_to(const std::string &filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing : " + filename);
    }
    out.write(reinterpret_cast<const char *>(buffer_), sizeof(T) * size_);
    out.close();
  }

  Tensor<T> reshape(const std::vector<size_t> &new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                      size_t(1), std::multiplies<size_t>());

    if (new_size != size_) {
      throw std::invalid_argument("Total size mismatch in reshape");
    }

    Tensor<T> reshaped = *this; // invoke copy constructor
    reshaped.shape_ = new_shape;
    return reshaped;
  }
  Tensor<T> transpose() const {
    if (shape_.size() != 2) {
      throw std::invalid_argument("transpose() only supports 2D tensors");
    }

    size_t rows = shape_[0];
    size_t cols = shape_[1];
    std::vector<T> transposed_data(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        transposed_data[j * rows + i] = buffer_[i * cols + j];
      }
    }
    return Tensor<T>({cols, rows}, std::move(transposed_data));
  }

  void flatten() { shape_ = {size_}; }
};

#endif
