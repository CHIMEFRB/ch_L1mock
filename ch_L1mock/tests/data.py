"""
Module for managing and generating test data.

"""


import os
import glob
from os import path

import numpy as np
from numpy import random
import h5py
import bitshuffle.h5 as bshufh5

from ..constants import ADC_SAMPLE_RATE, NSAMP_FPGA_FFT


# Constants
# ---------


TEST_DATA_DIR = path.join(path.dirname(__file__), 'data')
TEST_DATA_FILES = glob.glob(path.join(TEST_DATA_DIR, '*.h5'))


TIME_PER_PACKET = NSAMP_FPGA_FFT / ADC_SAMPLE_RATE
# Gives ~1.3ms cadence.  Needs to be multiple of 256: 16 for upchannelization
# and 16 for SSE square accumulation.
PACKETS_PER_INTEGRATION = 512

TIME_PER_INTEGRATION = TIME_PER_PACKET * PACKETS_PER_INTEGRATION

# Amount of test data.
TOTAL_TIME = 100.    # Approximate
NFREQ = 128
NPOL = 2

CHUNK = (64, 2, 256)
INTEGRATIONS_PER_FILE = CHUNK[2] * 64



def generate(outdir=None):
    """Generates test data.

    Data is just Gaussian random numbers with roughly the expected offset and
    amplitude.

    By default, data is generated in a subdirectory of the `tests` module. For
    system installations users will probably not have write permissions to this
    directory and thus gereration will fail. However, this script is run on
    installation, so users should not need to re-run it.

    """

    if not outdir:
        outdir = TEST_DATA_DIR

    # Generate some metadata.
    freq = ADC_SAMPLE_RATE - ADC_SAMPLE_RATE / NFREQ * np.arange(NFREQ)
    pol = ['XX', 'YY']

    # Set up paths.
    if not path.isdir(outdir):
        os.mkdir(outdir)

    # Some precomputed numbers.
    # Will simulate data one chunk at a time, for memory or in case we want to
    # do something more sophisticated eventually.
    this_ntime = CHUNK[2]
    nsamp_integration = int(round(TIME_PER_INTEGRATION
                                  * ADC_SAMPLE_RATE / NFREQ / 2))
    # In correator units, assuming RMS of 2 bits.
    # Note that this is an integer, which is realistic.
    Tsys = 16 * 2 * nsamp_integration

    integrations = 0
    while integrations * TIME_PER_INTEGRATION < TOTAL_TIME:
        if integrations % INTEGRATIONS_PER_FILE == 0:
            if integrations != 0:
                f.close()
            fname = '%08d.h5' % int(round(integrations * TIME_PER_INTEGRATION))
            fname = path.join(outdir, fname)
            f = h5py.File(fname, mode='w')
            # Index map
            im = f.create_group('index_map')
            im.create_dataset('pol', data=pol)
            im.create_dataset('freq', data=freq)
            time = im.create_dataset(
                    'time',
                    dtype=np.float64,
                    shape=(0,),
                    chunks=(CHUNK[2],),
                    maxshape=(INTEGRATIONS_PER_FILE,),
                    )
            # Main datasets
            intensity = f.create_dataset(
                    'intensity',
                    dtype=np.float32,
                    shape=(NFREQ, NPOL, 0),
                    chunks=CHUNK,
                    maxshape=(NFREQ, NPOL, INTEGRATIONS_PER_FILE),
                    compression=bshufh5.H5FILTER,
                    compression_opts=(0, bshufh5.H5_COMPRESS_LZ4),
                    )
            weight = f.create_dataset(
                    'weight',
                    dtype=np.uint8,
                    shape=(NFREQ, NPOL, 0),
                    chunks=CHUNK,
                    maxshape=(NFREQ, NPOL, INTEGRATIONS_PER_FILE),
                    compression=bshufh5.H5FILTER,
                    compression_opts=(0, bshufh5.H5_COMPRESS_LZ4),
                    )

        # Time axis.
        this_time = np.arange(integrations, integrations + this_ntime,
                              dtype=np.float64) * TIME_PER_INTEGRATION
        curr_ntime_file = time.shape[0]
        time.resize((curr_ntime_file + this_ntime,))
        time[curr_ntime_file:] = this_time
        # The data.
        this_intensity = np.empty((NFREQ, NPOL, this_ntime), dtype=np.float32)
        this_intensity[:] = Tsys
        this_noise = random.randn(*(this_intensity.shape))
        this_noise *= Tsys / np.sqrt(nsamp_integration)
        # Discretize.
        this_noise = np.round(this_noise)
        this_intensity += this_noise
        intensity.resize((NFREQ, NPOL, curr_ntime_file + this_ntime))
        intensity[:,:,curr_ntime_file:] = this_intensity
        # Weights
        weight.resize((NFREQ, NPOL, curr_ntime_file + this_ntime))
        # Full weight for now.
        weight[:,:,curr_ntime_file:] = 255

        integrations += this_ntime
    f.close()



