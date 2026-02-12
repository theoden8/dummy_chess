import os
import skbuild

print("CWD", os.path.abspath(os.curdir))

cmake_args = [
    "-G",
    "Unix Makefiles",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_PYTHON=ON",
    "-DBUILD_LIBRARIES=OFF",
    "-DBUILD_EXECUTABLES=OFF",
    "-DBUILD_CURSES=OFF",
    "-DOPTION_SUPPORT_JEMALLOC=disabled",
]

skbuild.setup(
    cmake_args=cmake_args,
    zip_safe=False,
    packages=["dummy_chess"],
    package_dir={"dummy_chess": "python/dummy_chess"},
    cmake_install_dir="python/dummy_chess",
    cmake_source_dir=".",
    include_package_data=False,
)
