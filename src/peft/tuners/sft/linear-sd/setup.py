import os
from setuptools import setup, Extension

from torch.utils import cpp_extension

#import torch_scatter
#ts_path = torch_scatter.__path__[0]

#include_dirs = [os.path.join(ts_path, 'csrc')]
#print(include_dirs)

extension = cpp_extension.CppExtension(
    'linear_sd_cpp',
    ['linear_sd.cpp'],
    #include_dirs=include_dirs,
)

setup(
    name='linear_sd_cpp',
    ext_modules=[extension],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
