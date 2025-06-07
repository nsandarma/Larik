from __future__ import annotations
from typing import List, Sequence, Tuple, Union

from numpy.testing import assert_allclose
from larik import _larik as _lr
from larik._dtype import Dtype
import time,math

class Larik:
  def __init__(self,buff,dtype:Dtype | str = Dtype.float32):
    if isinstance(dtype,str): dtype = Dtype.from_str(dtype)
    self._buffer = buff
    self._dtype = dtype

  def __str__(self): return f"<{self.dtype} {self.shape}>"
  def __repr__(self): return str(self)


  # Slicing
  def matmul(self, other: "Larik") -> "Larik":
    # Tentukan dtype hasil
    result_dtype = Dtype.promote_dtype(self.dtype, other.dtype)
    # Cast hanya jika perlu
    a = self if self.dtype == result_dtype else self.astype(result_dtype)
    b = other if other.dtype == result_dtype else other.astype(result_dtype)
    # Matmul
    result_buffer = a.buffer.matmul(b.buffer)
    return Larik(result_buffer, result_dtype)

  def __matmul__(self, other): return self.matmul(other)

  def __add__(self, other): 
    buff = self.buffer.add(other.buffer) if isinstance(other,Larik) else self.buffer.add_item(other)
    return Larik(buff,self.dtype)

  def __sub__(self, other): return Larik(self.buffer - other.buffer, self.dtype)

  def __getitem__(self,idx):
    if isinstance(idx,slice):
      return "is slice"
    return type(idx)

  # Attribute:
  @property
  def buffer(self):return self._buffer

  @property
  def dtype(self):return self._dtype

  @property
  def shape(self):return tuple(self.buffer.shape())

  @property
  def size(self):return self.buffer.size()

  @property
  def ndim(self):return self.buffer.dim()

  @property
  def T(self): return self.transpose()

  def astype(self,dtype:Dtype): return Larik(Dtype.cast(self._buffer,dtype),dtype)

  def flatten(self): return Larik(self.buffer.flatten(),self.dtype)

  
  def reshape(self,shape:Sequence):
    assert math.prod(shape) == self.size, "reshape is error !"
    return Larik(self.buffer.reshape(shape),dtype=self.dtype)
  
  def transpose(self):
    buff = self.buffer.transpose()
    return Larik(buff,dtype=self.dtype)

  def numpy(self):
    import numpy as np
    dtype = Dtype.to_numpy(self.dtype)
    return np.array(self._buffer.data(),dtype=dtype).reshape(self.shape)

  # def numpy(self):
  #   import numpy as np
  #   import ctypes
  #   np_dtype = Dtype.to_numpy(self.dtype.name)
  #   buffer_ptr = ctypes.cast(self._buffer.data(), ctypes.POINTER(np_dtype))
  #   data_ptr = np.ctypeslib.as_array(buffer_ptr, shape=(self.size,))
  #
  #   if hasattr(self, "strides"):
  #     # strides dalam bytes, NumPy butuh byte unit
  #     strides_bytes = tuple(s * np_dtype().itemsize for s in self.strides)
  #     return np.lib.stride_tricks.as_strided(data_ptr, shape=self.shape, strides=strides_bytes)
  #   else: return data_ptr.reshape(self.shape)

def create(data,dtype:Dtype | str = Dtype.float32) -> Larik:
  buff = Dtype.to_class(dtype).from_nested(data)
  return Larik(buff=buff,dtype=dtype)

def createND(data,shape:Sequence,dtype:Dtype | str =Dtype.float32) -> Larik:
  buff = Dtype.to_class(dtype)(shape,data)
  return Larik(buff=buff,dtype=dtype)

def arange(start,end,step):
  if any(isinstance(i,float) for i in (start,end,step)): dtype = Dtype.float32
  else: dtype = Dtype.int32
  buff = Dtype.to_class(dtype).arange(start,end,step)
  return Larik(buff=buff,dtype=dtype)

def zeros(shape:Sequence,dtype:Dtype=Dtype.int32) -> Larik:
  buff = Dtype.to_class(dtype).zeros(shape)
  return Larik(buff=buff,dtype=dtype)


def rand(shape:Sequence,seed:int=0,dtype:Dtype | str =Dtype.float32) -> Larik:
  buff = Dtype.to_class(dtype).rand(shape,seed)
  return Larik(buff=buff,dtype=dtype)

def randn(shape:Sequence,seed:int=0,dtype:Dtype | str =Dtype.float32) -> Larik:
  buff = Dtype.to_class(dtype).randn(shape,seed)
  return Larik(buff=buff,dtype=dtype)

def randint(shape:Sequence,low:int=0,high:int=100) -> Larik:
  buff = Dtype.to_class(Dtype.int32).randint(shape,low,high)
  return Larik(buff=buff,dtype=Dtype.int32)


