from setuptools import setup, Extension
import os

from Cython.Build import cythonize
import numpy as np

# Supplies __version__
from ch_L1mock import __version__


REQUIRES = ['numpy', 'scipy', 'cython', 'h5py', 'bitshuffle', 'burst_search']

# Generate test data.
from ch_L1mock.tests.data import generate
generate()

COMPILE_FLAGS = ['-O3', '-ffast-math', '-march=native']

EXTENSIONS = [
        Extension('ch_L1mock._L0',
            ['ch_L1mock/_L0.pyx'],
            extra_compile_args=COMPILE_FLAGS,
            language="c++",
            )
        ]


extensions = cythonize(EXTENSIONS,
        include_path=[np.get_include()],
        )

setup(
    name = 'ch_L1mock',
    version = __version__,
    packages = ['ch_L1mock', 'ch_L1mock.tests'],
    ext_modules = extensions,
    scripts=['bin/corr-vdif'],
    install_requires=REQUIRES,
    package_data = {'ch_L1mock.tests' : ['data/*']},

    author = "Kiyoshi Masui, Cherry Ng, Kendrick Smith",
    author_email = "kiyo@physics.ubc.ca",
    description = "Mock-up of CHIME FRB level 1 processing",
    url = "https://github.com/CHIMEFRB/ch_L1mock"
)
