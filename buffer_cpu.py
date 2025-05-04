from blas import libacc, CBLAS_COL_MAJOR, CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS
import ctypes,time,random
import numpy as np


class Array:
  _dtype_map = {
      'float32': ctypes.c_float,
      'float64': ctypes.c_double,
      'int32': ctypes.c_int,
      'int64': ctypes.c_long,
  }

  def _check_compatibility(self, other):
    if not isinstance(other, Array):
      raise TypeError("Requires Array instance")
    if len(self.shape) != 2 or len(other.shape) != 2:
      raise ValueError("Both arrays must be 2D")
    if self.shape[1] != other.shape[0]:
      raise ValueError(f"Shapes {self.shape} and {other.shape} incompatible")
    if self.dtype != other.dtype:
      raise TypeError("Dtype mismatch")

  def __init__(self, shape, dtype='float32'):
    self.shape = shape if isinstance(shape, tuple) else (shape,)
    self.size = 1
    for dim in self.shape:
      self.size *= dim
    self.dtype = dtype
    self._ctype = self._dtype_map[dtype]
    self._buffer = (self._ctype * self.size)()

  def __getitem__(self, idx):
    if isinstance(idx, tuple):
      if len(idx) != len(self.shape):
        raise IndexError("Incorrect number of indices")
      flat_index = self._ravel_index(idx)
    else:
      flat_index = idx
    return self._buffer[flat_index]

  def __setitem__(self, idx, value):
    if isinstance(idx, tuple):
      if len(idx) != len(self.shape):
        raise IndexError("Incorrect number of indices")
      flat_index = self._ravel_index(idx)
    else:
      flat_index = idx
    self._buffer[flat_index] = self._ctype(value)

  def _ravel_index(self, idx):
    flat_index = 0
    stride = 1
    for dim, i in zip(reversed(self.shape), reversed(idx)):
      flat_index += i * stride
      stride *= dim
    return flat_index

  def reshape(self, new_shape):
    new_size = 1
    for dim in new_shape:
      new_size *= dim
    if new_size != self.size:
      raise ValueError("Total size must remain unchanged")
    self.shape = new_shape
    return self  # for chaining

  def astype(self, new_dtype):
    new_array = Array(self.shape, dtype=new_dtype)
    for i in range(self.size):
      new_array._buffer[i] = self._dtype_map[new_dtype](self._buffer[i])
    return new_array

  def _tolist_recursive(self, idx, max_dim):
    if max_dim == 0:
      return self[idx]
    result = []
    for i in range(self.shape[len(idx)]):
      result.append(self._tolist_recursive(idx + (i,), max_dim - 1))
    return result

  def tolist(self): return self._tolist_recursive((), len(self.shape))

  @property
  def data(self): return ctypes.addressof(self._buffer)

  def data_as(self, ctype_ptr): return ctypes.cast(self.data, ctype_ptr)

  def __repr__(self): return f"Array(shape={self.shape}, dtype={self.dtype})"

  def __add__(self, other):
    if isinstance(other, Array):
      self._check_compatibility(other)
      result = self.astype(self.dtype)

      if self.dtype == 'float32':
        axpy = libacc.cblas_saxpy
        alpha = ctypes.c_float(1.0)
      else:
        axpy = libacc.cblas_daxpy
        alpha = ctypes.c_double(1.0)

      # BLAS: y = alpha*x + y
      axpy(self.size, alpha,
           other.data_as(ctypes.POINTER(other._ctype)), 1,
           result.data_as(ctypes.POINTER(result._ctype)), 1)
      return result

    # Array + scalar
    elif isinstance(other, (int, float)):
      result = self.astype(self.dtype)
      scalar = self._ctype(other)
      for i in range(self.size):
        result._buffer[i] += scalar
      return result
    else:
      return NotImplemented

  def __mul__(self, other):

    # Array * Array (element-wise)
    if isinstance(other, Array):
      self._check_compatibility(other)
      result = Array(self.shape, self.dtype)
      for i in range(self.size):
        result._buffer[i] = self._buffer[i] * other._buffer[i]
      return result

    # Array * scalar
    elif isinstance(other, (int, float)):
      result = self.astype(self.dtype)
      if self.dtype == 'float32':
        scal = libacc.cblas_sscal
        alpha = ctypes.c_float(other)
      else:
        scal = libacc.cblas_dscal
        alpha = ctypes.c_double(other)

      scal(self.size, alpha, result.data_as(ctypes.POINTER(result._ctype)), 1)
      return result
    else:
      return NotImplemented

  def __neg__(self):
    result = self.astype(self.dtype)
    if self.dtype == 'float32':
      scal = libacc.cblas_sscal
      alpha = ctypes.c_float(-1.0)
    else:
      scal = libacc.cblas_dscal
      alpha = ctypes.c_double(-1.0)
    # BLAS: x = -1*x
    scal(self.size, alpha, result.data_as(ctypes.POINTER(result._ctype)), 1)
    return result

  def __sub__(self, other):
    """Element-wise subtraction"""
    # Array - Array
    if isinstance(other, Array):
      self._check_compatibility(other)
      result = self.astype(self.dtype)

      if self.dtype == 'float32':
        axpy = libacc.cblas_saxpy
        alpha = ctypes.c_float(-1.0)
      else:
        axpy = libacc.cblas_daxpy
        alpha = ctypes.c_double(-1.0)

      axpy(self.size, alpha,
           other.data_as(ctypes.POINTER(other._ctype)), 1,
           result.data_as(ctypes.POINTER(result._ctype)), 1)
      return result

    # Array - scalar (optimized using BLAS)
    elif isinstance(other, (int, float)):
      result = self.astype(self.dtype)
      scalar = self._ctype(-other)  # Convert to negative upfront

      if self.dtype == 'float32':
        axpy = libacc.cblas_saxpy
        alpha = ctypes.c_float(1.0)
      else:
        axpy = libacc.cblas_daxpy
        alpha = ctypes.c_double(1.0)

      # Create temporary scalar array
      scalar_arr = (self._ctype * 1)(scalar)

      # BLAS: result = 1.0*scalar + self
      axpy(self.size, alpha,
           scalar_arr, 0,  # incX=0 for constant value
           result.ctypes.data_as(ctypes.POINTER(result._ctype)), 1)
      return result

    else:
      return NotImplemented

  def __matmul__(self, other):
    self._check_compatibility(other)
    """Matrix multiplication using BLAS GEMM"""
    M, K = self.shape
    _, N = other.shape

    # Create output array
    out = Array((M, N), dtype=self.dtype)

    # Get BLAS function based on dtype
    if self.dtype == 'float32':
      gemm = libacc.cblas_sgemm
      alpha = ctypes.c_float(1.0)
      beta = ctypes.c_float(0.0)
    else:
      gemm = libacc.cblas_dgemm
      alpha = ctypes.c_double(1.0)
      beta = ctypes.c_double(0.0)

    # Get raw pointers
    a_ptr = self.data_as(ctypes.POINTER(self._ctype))
    b_ptr = other.data_as(ctypes.POINTER(self._ctype))
    c_ptr = out.data_as(ctypes.POINTER(out._ctype))

    gemm(
        CBLAS_ROW_MAJOR,  # Row-major ordering
        CBLAS_NO_TRANS,   # No transpose A
        CBLAS_NO_TRANS,   # No transpose B
        M, N, K,          # Dimensions
        alpha,            # Scalar multiplier
        a_ptr, K,         # A matrix + leading dimension
        b_ptr, N,         # B matrix + leading dimension
        beta,             # Initial value multiplier
        c_ptr, N          # C matrix + leading dimension
    )

    return out
  
  def definitelyNotDot(self,other): return self.__matmul__(other)

  @staticmethod
  def rand(shape, dtype='float32'):
    """Create array with random values between 0 and 1"""
    if dtype not in ('float32', 'float64'):
      raise ValueError("rand() only supports float32/float64 dtypes")
    arr = Array(shape, dtype)
    for i in range(arr.size):
      arr._buffer[i] = random.random()  # Auto-converts to c_float/c_double
    return arr

  def numpy(self):
    """Zero-copy conversion to NumPy array using the existing buffer"""
    ctype_ptr = ctypes.cast(self._buffer, ctypes.POINTER(self._ctype))
    np_arr = np.ctypeslib.as_array(
        ctype_ptr,
        shape=self.shape
    )
    return np.asarray(np_arr, dtype=self.dtype)
  
  @staticmethod
  def zeros(shape):
    arr = Array(shape,"float64")
    for i in range(arr.size):
      arr._buffer[i] = 0.0
    return arr

  @staticmethod
  def ones(shape,dtype="float64"):
    arr = Array(shape,dtype)
    val = 1 if dtype in ["int32","int64"] else 1.0
    for i in range(arr.size):
      arr._buffer[i] = val
    return arr


if __name__ == "__main__":
  n = 1024
  arr = Array.ones((1024,1024),"int32")

