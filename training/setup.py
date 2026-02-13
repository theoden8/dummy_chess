"""Setup script for dummy-chess-training with Cython extensions."""

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "fast_features",
        sources=["fast_features.pyx"],
        include_dirs=[numpy.get_include(), ".", ".."],  # Parent dir for FEN.hpp
        extra_compile_args=["-O3", "-std=c++20"],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        build_dir="build/cython",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
