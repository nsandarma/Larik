import tensorlib
import numpy as np
import time
N = 2048


def matmul_int():
  a = tensorlib.TensorInt32.randint((N,N))
  b = tensorlib.TensorInt32.randint((N,N))
  a_numpy = np.array(a.data(),dtype=np.int32).reshape(N,N)
  b_numpy = np.array(b.data(),dtype=np.int32).reshape(N,N)
  tic = time.monotonic()
  c = tensorlib.TensorInt32.matmul_int(a,b)
  toc = time.monotonic()
  print(f"times [mylib] : {toc-tic}")

  tic = time.monotonic()
  c_numpy = a_numpy @ b_numpy
  toc = time.monotonic()
  print(f"times [mylib] : {toc-tic}")

  print(c_numpy)
  print()
  c_ = np.array(c.data(),dtype=np.int32).reshape(N,N)

  print(c_)
  np.testing.assert_allclose(c_,c_numpy)


def matmul_float():
  a = tensorlib.TensorFloat32.randn((N,N))
  b = tensorlib.TensorFloat32.randn((N,N))
  a_numpy = np.array(a.data(),dtype=np.float32).reshape(N,N)
  b_numpy = np.array(b.data(),dtype=np.float32).reshape(N,N)
  tic = time.monotonic()
  c = tensorlib.TensorFloat32.matmul_float(a,b)
  toc = time.monotonic()
  print(f"times [mylib] : {toc-tic}")

  tic = time.monotonic()
  c_numpy = a_numpy @ b_numpy
  toc = time.monotonic()
  print(f"times [mylib] : {toc-tic}")

  print(c_numpy)
  print()
  print(np.array(c.data(),dtype=np.float32).reshape(N,N))
  print(c)

def vector():pass


def test_memoryview():
  a = tensorlib.TensorInt32.randint([3, 3])
  m = memoryview(a.data())  # view C++ buffer

  del a  # kalau buffer dihapus dan Python akses m => crash
  print(m[0])  # harus tidak segfault

if __name__ == "__main__":
  test_memoryview()
