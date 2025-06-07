from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "tensorlib",
        ["src/bindings.cc"],
        include_dirs=[
            pybind11.get_include(),         # Header pybind11
            "include/tensor.hpp",                            # Header tensor.hpp
        ],
        language="c++",
        extra_compile_args=["-std=c++17","-DACCELERATE_NEW_LAPACK"]
        
    )
]

setup(
    name="tensorlib",
    version="1.0",
    ext_modules=ext_modules,
)
