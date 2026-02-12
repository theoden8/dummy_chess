import os
import pathlib

import setuptools
import torch.utils.cpp_extension

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

# setuptools requires relative paths from setup.py directory
REL_ROOT = os.path.relpath(ROOT_DIR, pathlib.Path(__file__).resolve().parent)

ext_modules = [
    torch.utils.cpp_extension.CppExtension(
        name="_preprocess",
        sources=[
            os.path.join("csrc", "preprocess.cpp"),
            os.path.join(REL_ROOT, "m42.cpp"),
        ],
        include_dirs=[str(ROOT_DIR)],
        extra_compile_args=[
            "-std=c++20",
            "-O3",
            "-march=native",
            "-ffast-math",
            "-DUSE_INTRIN",
            "-pthread",
        ],
        extra_link_args=["-pthread"],
    ),
]

# Only build CUDA extension if nvcc is available
if torch.utils.cpp_extension.CUDA_HOME is not None:
    ext_modules.append(
        torch.utils.cpp_extension.CUDAExtension(
            name="_extract_cuda",
            sources=[
                os.path.join("csrc", "extract_cuda.cu"),
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",  # Ada (RTX 4090)
                ],
            },
        ),
    )

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
