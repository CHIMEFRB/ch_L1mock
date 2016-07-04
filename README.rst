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
cases these will be installed automatically. It also depends on ch_frb_io,
Bitshuffle and Burst Search, which will not be installed automatically. However
these can be fetched via the requirements.txt file (see below).


.. _`ch_vdif_assembler`: https://github.com/kmsmith137/ch_vdif_assembler


Installation
------------

After cloning the repository, enter the repository directory. To install the
dependencies::

    pip install -r requirements.txt

To install this package::

    python setup.py install

or::

    python setup.py develop
    

Running the L1 Mock
-------------------

Installing this package will add two executables to your path: ``corr-vdif``
and ``run-L1-mock``. The former transforms baseband vdif data from a variety of
sources to intensity data. The latter, ``run-L1-mock``, runs the L1 mock.  The
program accepts a single parameter, a YAML config file. Example configurations
are available in the 'examples' directory.


Current Functionality Overview
------------------------------

As a rough overview of the current functionality, the L1 mock can:

- Accept data in baseband format from a variety of sources: disk, network,
  simulated.
- Accept data as correlated intensity from disk (HDF5 files).
- Noise-weight data data based on the radiometer equation.
- Preprocess data using standard routines from the Burst Search package,
  including de-trending and RFI identification.
- Inject simulated FRB events into the data stream with a variety of parameters.
- Perform the dispersion-measure transform on the data using Bonsai, with searches
  over spectral-index and scattering measure.
- Coarse grain SNR values of DM-transformed data.
- Identify events based on a simple threshold.
- Print events to STDOUT.
- Multi-thread: the DM transform is internally multi-threaded and can occur in
  parallel with preprocessing of input data or post-processing of output triggers.

An incomplete list of functionality not yet implemented, but could be without
too much hassle:

- Spectral up-sampling of baseband data to higher frequency resolution 
  (beyond 1024 channels).
- Semi-coherent dedispersion.
- Any kind of calibration beyond what the correlator provides. In particular
  the bandpass calibration is order unity incorrect, which severely hinders
  sensitivity.
- Accepting *intensity* data over the network.
- Any kind of grouping of triggers. Single input events currently generate
  dozens of output events.
- Detailed description of trigger events. Currently only the signal-to-noise ratio
  is printed to STDOUT.
- Accurate propagation of time stamps. Timing information is currently lost.

Permanent limitations that this package does not aim to overcome:

- Python parallelism: While dedispersion, coarse graining, and baseband packet
  assembly are all internally parallel, can occur in parallel with each other,
  and in parallel with a single python thread, the python threads themselves
  are constrained to run one at a time. This includes input-data preprocessing,
  coarse-grained trigger post-processing, acting on triggered events, and a
  likely-non-negligible amount of control overhead. Note however it is straight
  forward to have computationally intensive code (say in the preprocessing)
  switch to a compiled language briefly and allow the other python threads to
  proceed while the work is occurring.
