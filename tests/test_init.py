#!/usr/bin/env python3.11
import math,unittest,time,os
from larik import larik as lr
import numpy as np

from larik._dtype import Dtype
from larik.helpers import get_ratio_time

class TestLarik(unittest.TestCase):
  def setUp(self):
    a = [[ 1.0, 2.0, 3.0, 4.0],
          [ 5.0, 6.0, 7.0, 8.0],
          [ 9.0,10.0,11.0,12.0],
          [13.0,14.0,15.0,16.0]]
    b = [1,2,3,4,5,6,7,8,9,10]
    self.a = a
    self.b = b
    self.la = lr.create(a)
    self.lb = lr.createND(shape=(5,2),data=b)
    self.na = np.array(a,dtype=self.la.dtype.name)
    self.nb = np.array(b,dtype=self.lb.dtype.name).reshape(self.lb.shape)


  def test_init(self):
    self.assertTupleEqual(self.la.shape,self.na.shape)
    self.assertEqual(self.la.ndim,self.na.ndim)
    self.assertEqual(self.la.size,self.na.size)

    np.testing.assert_allclose(self.la.numpy(),self.na)
    np.testing.assert_allclose(self.lb.numpy(),self.nb)

    self.assertEqual(self.la.dtype.name,self.na.dtype.name)
    self.assertEqual(self.lb.dtype.name,self.nb.dtype.name)

  
  def test_reshape(self):
    new_shape = (self.la.size // 2,2)
    la = self.la.reshape(new_shape)
    na = self.na.reshape(new_shape)
    self.assertTupleEqual(la.shape,na.shape)
    self.assertEqual(la.size,na.size)
    self.assertEqual(la.ndim,na.ndim)
    np.testing.assert_allclose(la.numpy(),na)


  def test_transpose(self):
    la = self.la.transpose()
    na = self.na.transpose()
    self.assertTupleEqual(la.shape,na.shape)
    self.assertEqual(la.size,na.size)
    self.assertEqual(la.ndim,na.ndim)
    np.testing.assert_allclose(la.numpy(),na)

    if os.environ.get("WITH_TIME"):
      """
      without strides is no matter :)
      times  [larik]: 0.2244707079953514
      times [numpy]: 0.44166300000506453
      """
      N = 4084
      la = lr.rand((N,N))
      tic = time.monotonic()
      la_t = la.transpose()
      lx = la_t @ la_t
      print("TRANSPOSE : ")
      print(f"times  [larik]: {time.monotonic()-tic}")
      na = la.numpy()
      tic = time.monotonic()
      na = na.T
      nx = na @ na
      print(f"times [numpy]: {time.monotonic()-tic}")
      np.testing.assert_allclose(la_t.numpy(),na)
      np.testing.assert_allclose(lx.numpy(),nx,rtol=1e-5)



  def test_random(self):
    N = 1024
    tic = time.monotonic()
    la = lr.rand((N,N))
    tl = time.monotonic() - tic

    tic = time.monotonic()
    na= np.random.rand(N,N)
    tn = time.monotonic() - tic
    get_ratio_time(tl,tn)

    self.assertTupleEqual((N,N),la.shape)
    self.assertEqual(math.prod((N,N)),la.size)
    self.assertEqual(2,la.ndim)
    self.assertEqual(la.dtype,Dtype.float32)

    la = lr.randn((N,N))
    self.assertTupleEqual((N,N),la.shape)
    self.assertEqual(math.prod((N,N)),la.size)
    self.assertEqual(2,la.ndim)
    self.assertEqual(la.dtype,Dtype.float32)

    la = lr.randint((N,N))
    self.assertTupleEqual((N,N),la.shape)
    self.assertEqual(math.prod((N,N)),la.size)
    self.assertEqual(2,la.ndim)
    self.assertEqual(la.dtype,Dtype.int32)

  def test_str(self):
    print(self.la)


if __name__ == '__main__':
  unittest.main()

