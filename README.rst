=================
CHIME FRB L1 Mock
=================

Mock-up of CHIME FRB level 1 processing, for development and early searches.


Dependancies
------------

This package depends on the `ch_vdif_assembler`_ C++ and Python library. This
has a non-Python standard installation proceedure and must be installed manually.
This dependancy is only required for generating input streams (the `L0` module).
If you will be reading intensity data from hdf5 files it is not required.

It also depends on the Numpy, Scipy, Cython and h5py Python packages. In most
cases these will be installed automatically if using pip.

.. _`ch_vdif_assembler`: https://github.com/kmsmith137/ch_vdif_assembler


Installation
------------

To install, clone the repository and::

    python setup.py install


