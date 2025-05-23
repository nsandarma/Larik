#!/usr/bin/env python3.11
import unittest,time
from buffer_cpu import Array
import numpy as np,time

"""
"""

class TestBufferCPU(unittest.TestCase):
  def setUp(self) -> None:
    N = 1024
    self.a = Array.rand((N,N)).astype("float32")
    self.b = Array.rand((N,N)).astype("float32")
    self.a_numpy = self.a.numpy()
    self.b_numpy = self.b.numpy()
  
  def check_type(self):
    print(self.a.dtype)
    print(self.b.dtype)
    print(self.a_numpy.dtype)
    print(self.b_numpy.dtype)

  def test_matmul(self):
    c = self.a @ self.b
    c_numpy = self.a_numpy @ self.b_numpy
    np.testing.assert_allclose(c.numpy(),c_numpy,rtol=1e-5,atol=1e-8)

  def test_add(self):
    c = self.a + self.b
    c_numpy = self.a_numpy + self.b_numpy
    np.testing.assert_allclose(c.numpy(),c_numpy,rtol=1e-5,atol=1e-8)

  def test_substract(self):
    c = self.a - self.b
    c_numpy = self.a_numpy - self.b_numpy
    np.testing.assert_allclose(c.numpy(),c_numpy,rtol=1e-5,atol=1e-8)

  def test_mul(self):
    c = self.a * self.b
    c_numpy = self.a_numpy * self.b_numpy
    np.testing.assert_allclose(c.numpy(),c_numpy,rtol=1e-5,atol=1e-8)

  def test_time(self):
    t1 = time.monotonic()
    c = self.a @ self.b
    t2 =  time.monotonic()
    c_numpy = self.a_numpy @ self.b_numpy
    t3 = time.monotonic()
    my_lib = t2 - t1
    numpy = t3-t2
    self.assertLessEqual(my_lib,numpy)



if __name__ == "__main__":
  start = time.monotonic()
  unittest.main(exit=False)
  end = time.monotonic()
  print(f"Total test time: {end - start:.6f} seconds")

