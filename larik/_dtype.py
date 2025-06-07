from enum import Enum,auto
from larik import _larik as _lr


class Dtype(Enum):
  int8 = auto(); uint8 = auto(); uint16 = auto(); int16 = auto(); int32 = auto()
  uint32 = auto(); uint64 = auto(); int64 = auto(); float32 = auto(); float64 = auto()

  @staticmethod
  def to_class(dtype):
    if isinstance(dtype,str): dtype = Dtype.from_str(dtype)
    mapping = {
      Dtype.int8: _lr.Tensorint8, Dtype.uint8: _lr.Tensoruint8,
      Dtype.int16: _lr.Tensorint16, Dtype.uint16: _lr.Tensoruint16,
      Dtype.int32: _lr.Tensorint32, Dtype.uint32: _lr.Tensoruint32,
      Dtype.int64: _lr.Tensorint64, Dtype.uint64: _lr.Tensoruint64,
      Dtype.float32: _lr.Tensorfloat32, Dtype.float64: _lr.Tensorfloat64,
    }
    return mapping[dtype]
  
  @staticmethod
  def cast(buff, dtype):
    mapping = {
        dtype.int8: lambda: buff.cast_to_int8(),
        dtype.uint8: lambda: buff.cast_to_uint8(),
        dtype.int16: lambda: buff.cast_to_int16(),
        dtype.uint16: lambda: buff.cast_to_uint16(),
        dtype.int32: lambda: buff.cast_to_int32(),
        dtype.uint32: lambda: buff.cast_to_uint32(),
        dtype.int64: lambda: buff.cast_to_int64(),
        dtype.uint64: lambda: buff.cast_to_uint64(),
        dtype.float32: lambda: buff.cast_to_float32(),
        dtype.float64: lambda: buff.cast_to_float64(),
    }
    if dtype not in mapping: raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]()  # panggil lambda

  @staticmethod
  def promote_dtype(dtype1, dtype2):
    """Promote dtype seperti NumPy"""
    priority = {
      Dtype.int8: 0, Dtype.uint8: 1, Dtype.int16: 2, Dtype.uint16: 3,
      Dtype.int32: 4, Dtype.uint32: 5, Dtype.int64: 6, Dtype.uint64: 7,
      Dtype.float32: 8, Dtype.float64: 9,
    }
    return dtype1 if priority[dtype1] >= priority[dtype2] else dtype2
  
  @staticmethod
  def to_numpy(dtype):
    import numpy as np
    mapping = {
      dtype.int8: np.int8, dtype.uint8: np.uint8, dtype.int16: np.int16,
      dtype.uint16: np.uint16, dtype.int32: np.int32, dtype.uint32: np.uint32,
      dtype.int64: np.int64, dtype.uint64: np.uint64, dtype.float32: np.float32,
      dtype.float64: np.float64
    }
    return mapping[dtype]

  @staticmethod
  def from_numpy(dtype):
    import numpy as np
    mapping = {
      np.int8: dtype.int8, np.uint8: dtype.uint8, np.int16: dtype.int16,
      np.uint16: dtype.uint16, np.int32: dtype.int32, np.uint32: dtype.uint32,
      np.int64: dtype.int64, np.uint64: dtype.uint64, np.float32: dtype.float32,
      np.float64: dtype.float64
    }
    return mapping[dtype]
  
  @staticmethod
  def from_str(dtype):
    mapping = {
      "int8": Dtype.int8, "uint8": Dtype.uint8, "int16": Dtype.int16,
      "uint16": Dtype.uint16, "int32": Dtype.int32, "uint32": Dtype.uint32,
      "int64": Dtype.int64, "uint64": Dtype.uint64, "float32": Dtype.float32,
      "float64": Dtype.float64,
    }
    return mapping[dtype]


if __name__ == "__main__":
  print(Dtype.int16)
