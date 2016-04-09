from setuptools import setup
import os

# Supplies __version__
from ch_L1mock import __version__


REQUIRES = ['numpy', 'scipy', 'cython', 'h5py', 'bitshuffle']

# Generate test data.
from ch_L1mock.tests.data import generate
generate()

setup(
    name = 'ch_L1mock',
    version = __version__,
    packages = ['ch_L1mock', 'ch_L1mock.tests'],
    scripts=[],
    install_requires=REQUIRES,
    pacakge_data = {'ch_L1mock.tests' : 'data/*'},

    author = "Kiyoshi Masui, Cherry Ng, Kendrick Smith",
    author_email = "kiyo@physics.ubc.ca",
    description = "Mock-up of CHIME FRB level 1 processing",
    url = "https://github.com/CHIMEFRB/ch_L1mock"
)
