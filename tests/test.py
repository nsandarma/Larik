import tinygrad as tn
import numpy as np

if __name__ == "__main__":
  a = tn.Tensor.rand(2,3).realize()
  b = tn.Tensor.rand(3,2).realize()
  c = a.matmul(b)
  print(c.contiguous())


