from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import  build_ext

# Path ke include dan source
include_dirs = [
    pybind11.get_include(),
    "./_core/include/tensor.hpp"
]

ext_modules = [
    Extension(
        name="larik._larik",  # ⬅️ output langsung ke larik/
        sources=["./_core/src/bindings.cc"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17","-DACCELERATE_NEW_LAPACK"]
    ),
]

setup(
  name="larik",
  version="1.0",
  packages=["larik"],
  ext_modules=ext_modules,
  cmdclass={"build_ext":build_ext}
)

