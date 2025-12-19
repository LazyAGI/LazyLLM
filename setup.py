import os
import sys
import platform
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='csrc'):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is required to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lazyllm'))
        cfg = 'Debug' if self.debug else 'Release'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        build_args = []

        if platform.system() == "Windows":
            cmake_args += ['-A', 'x64']
            build_args += ['--config', cfg]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name='lazyllm_cpp',
    version='0.7.0',
    author='wangzhihong',
    author_email='wangzhihong@sensetime.com',
    description='LazyLLM C++ extension with pybind11',
    packages=find_packages(),
    ext_modules=[CMakeExtension('lazyllm_cpp', sourcedir='csrc')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
