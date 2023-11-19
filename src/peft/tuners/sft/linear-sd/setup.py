import os
from setuptools import setup, Extension

from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

#import torch_scatter
#ts_path = torch_scatter.__path__[0]

#include_dirs = [os.path.join(ts_path, 'csrc')]
#print(include_dirs)

extension = CUDAExtension(
    'linear_sd_cpp',
    ['linear_sd.cpp', 'linear_sd_cuda.cu'],
    #include_dirs=include_dirs,
    extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
)

setup(
    name='linear_sd_cpp',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
)
