"""Build script for Cython fast_features module."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "fast_features",
        sources=["fast_features.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="fast_features",
    ext_modules=cythonize(extensions, language_level=3),
)
