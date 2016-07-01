=================
CHIME FRB L1 Mock
=================

Mock-up of CHIME FRB level 1 processing, for development and early searches.


Dependancies
------------

This package depends on the `ch_vdif_assembler`_ C++ and Python library. This
has a non-Python standard installation procedure and must be installed manually.
This dependency is only required for generating input streams (the `L0` module).
If you will be reading intensity data from hdf5 files it is not required.

It also depends on the Numpy, Scipy, Cython and h5py Python packages. In most
cases these will be installed automatically. It also depends on ch_frb_io, Bitshuffle 
and Burst Search, which will not be installed automatically. However these can be
fetched via the requirements.txt file (see below).


.. _`ch_vdif_assembler`: https://github.com/kmsmith137/ch_vdif_assembler


Installation
------------

After cloning the repository, enter the repository directory. To install the
dependancies::

    pip install -r requirements.txt

To install this package::

    python setup.py install

or::

    python setup.py develop
    

Running the L1 Mock
-------------------

Installing this package will add the executable ``run-L1-mock`` to your path. The
program accepts a single parameter, a YAML config file. Example configuration are
available in the 'examples' directory.
