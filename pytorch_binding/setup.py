from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import platform

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')

def make_relative_rpath(path):
    
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        return ''
    else:
        return '-Wl,-rpath,$ORIGIN/' + path



setup(
    name='warpctc',
    ext_modules=[
        # apparently pybind does not support submodules like warpctc._warpctc
        CppExtension('_warpctc', ['src/warpctc.cpp'],
        include_dirs=['/home/tv/data/pytorch-deep-speech/warp-ctc/include'],
        library_dirs=['/home/tv/data/pytorch-deep-speech/warp-ctc/build'],
        libraries=['warpctc'],
        extra_link_args=[make_relative_rpath('warpctc')])
    ],
    packages=find_packages(exclude=['tests']),
    package_data={'warpctc':['libwarpctc.so']},
    
    cmdclass={
        'build_ext': BuildExtension
    })
