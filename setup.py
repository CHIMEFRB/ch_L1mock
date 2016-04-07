from setuptools import setup
import os

from ch_L1mock import __version__


REQUIRES = ['numpy', 'scipy', 'cython', 'h5py',]

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES

setup(
    name = 'ch_L1mock',
    version = __version__,
    packages = ['ch_L1mock', 'ch_L1mock.tests'],
    scripts=[],
    install_requires=requires,

    author = "Kiyoshi Masui, Cherry Ng, Kendrick Smith",
    author_email = "kiyo@physics.ubc.ca",
    description = "Mock-up of CHIME FRB level 1 processing",
    url = "https://github.com/CHIMEFRB/ch_L1mock"
)
