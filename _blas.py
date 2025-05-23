import ctypes,time
import ctypes.util
import numpy as np
import os,time

if os.environ.get("BLAS") == "1":
  blas_path = "blas/lib/libopenblas.dylib"
  print("USING OPENBLAS ...")
else: blas_path = ctypes.util.find_library("libblas.dylib")

libacc = ctypes.CDLL(blas_path)

# Constanta
CBLAS_ROW_MAJOR = 101
CBLAS_COL_MAJOR = 102
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112

# 3. Set argumen untuk cblas_dgemm
## Configure both single and double precision GEMM
for dtype, blas_type in [('float32', ctypes.c_float), ('float64', ctypes.c_double)]:
  libacc_name = f'cblas_{"s" if dtype == "float32" else "d"}gemm'
  func = getattr(libacc, libacc_name)
  func.argtypes = [
    ctypes.c_int,         # Order
    ctypes.c_int,         # TransA
    ctypes.c_int,         # TransB
    ctypes.c_int,         # M
    ctypes.c_int,         # N
    ctypes.c_int,         # K
    blas_type,            # alpha
    ctypes.POINTER(blas_type),  # A
    ctypes.c_int,         # lda
    ctypes.POINTER(blas_type),  # B
    ctypes.c_int,         # ldb
    blas_type,            # beta
    ctypes.POINTER(blas_type),  # C
    ctypes.c_int          # ldc
  ]
  func.restype = None


if __name__ == "__main__":
  N = 1024
  A = np.random.rand(N,N).astype(np.double)
  B = np.random.rand(N,N).astype(np.double)
  C_blas = np.zeros((N,N),dtype=np.double)

  A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  C_blasptr = C_blas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  tic = time.monotonic()
  libacc.cblas_dgemm(
    CBLAS_ROW_MAJOR,CBLAS_NO_TRANS,CBLAS_NO_TRANS,
    N,N,N,1.0,A_ptr,N,B_ptr,N,0.0,C_blasptr,N)
  print(f"times : {time.monotonic()-tic}")

  tic = time.monotonic()
  print(C_blas)
  print("-"*100)
  print(A @ B)
  print(f"times : {time.monotonic()-tic}")
