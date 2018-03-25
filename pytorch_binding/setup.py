# WarpCTC PyTorch bindings (c) 2018 by Thomas Viehmann <tv@lernapparat.de>
# All rights reserved. Licensed under the
# Apache License,  Version 2.0, January 2004
# see LICENSE in root directory

from setuptools import setup, find_packages, Command, distutils
from torch.utils.cpp_extension import BuildExtension, CppExtension
import platform
import os
import subprocess
import shutil
import distutils.command.clean
import distutils.command.build

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
    res = subprocess.run(['cmake', '../../../'], cwd='build/lib')
    assert res.returncode == 0, "Error"
    res = subprocess.run(['make'], cwd='build/lib')
    assert res.returncode == 0, "Error"
    os.makedirs('warpctc/lib', exist_ok=True)
    shutil.copy('build/lib/libwarpctc.so', 'warpctc/lib/')


class Clean(distutils.command.clean.clean):

    def run(self):
        import glob
        remove = ['build', 'dist', '*.egg-info', 'warpctc/lib',
                  'warpctc/__pycache__']
        for wildcard in remove:
            for filename in glob.glob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


class BuildDeps(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_warpctc_so()


class Build(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands


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
    package_data={'warpctc': ['lib/libwarpctc.so']},

    cmdclass={
        'build': Build,
        'build_deps': BuildDeps,
        'build_ext': BuildExtension,
        'clean': Clean,
    })
