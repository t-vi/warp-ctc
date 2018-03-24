from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import platform
import os
import subprocess, shutil

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')

def make_relative_rpath(path):
    
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        return ''
    else:
        return '-Wl,-rpath,$ORIGIN/' + path


def build_warpctc_so():
    os.makedirs('build/lib', exist_ok=True)
    res = subprocess.run(['cmake','../../../'], cwd='build/lib')
    assert res.returncode == 0, "Error"
    res = subprocess.run(['make'], cwd='build/lib')
    assert res.returncode == 0, "Error"    
    os.makedirs('warpctc/lib', exist_ok=True)
    shutil.copy('build/lib/libwarpctc.so','warpctc/lib/')

build_warpctc_so()

setup(
    name='warpctc',
    ext_modules=[
        # apparently pybind does not support submodules like warpctc._warpctc
        CppExtension('warpctc._warpctc', ['src/_warpctc.cpp'],
        include_dirs=['../include'],
        library_dirs=['build/lib'],
        libraries=['warpctc'],
        extra_link_args=[make_relative_rpath('lib')])
    ],
    packages=find_packages(exclude=['tests']),
    package_data={'warpctc':['lib/libwarpctc.so']},
    
    cmdclass={
        'build_ext': BuildExtension
    })
