import subprocess
import os
import sys

import setuptools
import skbuild

print('CWD', os.path.abspath(os.curdir))

skbuild.setup(
    cmake_args=[
        "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
    ],
    zip_safe=False,
    packages=['dummy_chess'],
    package_dir={"dummy_chess": "python/dummy_chess"},
    cmake_install_dir="python",
    cmake_source_dir="python",
)
