from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
  Extension("linear_global", language="c", sources=["linear_global.pyx"], extra_compile_args=[
    "-DNPY_NO_DEPRECATED_API NPY_1_7_API_VERSION",
    "-Wno-unreachable-code-fallthrough", 
    "-Wno-unreachable-code"
  ])
]

setup(
  ext_modules = cythonize(ext_modules, compiler_directives={ 'language_level': '3' }),
  include_dirs=[numpy.get_include()]                
)
