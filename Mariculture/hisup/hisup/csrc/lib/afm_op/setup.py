# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import glob
# import os
#
# extension_dir = '.'
#
# main_file = glob.glob(os.path.join(extension_dir,'*.cpp'))
# source_cuda = glob.glob(os.path.join(extension_dir,'cuda',
# '*.cu'))
#
# sources = main_file + source_cuda
#
# extra_compile_args = {'cxx': []}
# defined_macros = []
# extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]
#
# extension = CUDAExtension
#
# include_dirs = [extension_dir]
#
# ext_module = [
#     extension(
#         "CUDA",
#         sources,
#         include_dirs=include_dirs,
#         defined_macros=defined_macros,
#         # extra_compile_args=extra_compile_args,
#     )
# ]
# setup(
#     name='afm_op',
#     ext_modules=ext_module,
#     cmdclass={
#         'build_ext': BuildExtension
#     })
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

extension_dir = '.'

main_file = glob.glob(os.path.join(extension_dir, '*.cpp'))
source_cuda = glob.glob(os.path.join(extension_dir, 'cuda', '*.cu'))

sources = main_file + source_cuda

# 定义编译参数
extra_compile_args = {
    'cxx': [],
    'nvcc': [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
"-gencode", "arch=compute_89,code=sm_89"
    ]
}

# 定义宏
defined_macros = []

# 包含目录
include_dirs = [extension_dir]

# 创建 CUDAExtension
ext_module = [
    CUDAExtension(
        name="afm_op",
        sources=sources,
        include_dirs=include_dirs,
        defined_macros=defined_macros,
        extra_compile_args=extra_compile_args
    )
]

# 设置
setup(
    name='afm_op',
    ext_modules=ext_module,
    cmdclass={
        'build_ext': BuildExtension
    }
)