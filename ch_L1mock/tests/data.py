"""
Module for managing and generating test data.

"""

from os import path
import glob

import numpy as np
from numpy import random

from ..constants import ADC_SAMPLE_RATE, NSAMP_FPGA_FFT
from .. import io


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


def generate(outdir=None):
    """Generates test data.

    Data is just Gaussian random numbers with roughly the expected offset and
    amplitude.

    By default, data is generated in a subdirectory of the `tests` module. For
    system installations users will probably not have write permissions to this
    directory and thus generation will fail. However, this script is run on
    installation, so users should not need to re-run it.

    """

    if not outdir:
        outdir = TEST_DATA_DIR

    # Generate some meta-data.
    freq = ADC_SAMPLE_RATE - ADC_SAMPLE_RATE / NFREQ * np.arange(NFREQ)
    pol = ['XX', 'YY']

    # Set up paths.

    # Some precomputed numbers.
    # Will simulate data one chunk at a time, for memory or in case we want to
    # do something more sophisticated eventually.
    nsamp_integration = int(round(TIME_PER_INTEGRATION
                                  * ADC_SAMPLE_RATE / NFREQ / 2))
    # In correlator units, assuming RMS of 2 bits.
    # Note that this is an integer, which is realistic.
    Tsys = 16 * 2 * nsamp_integration

    writer = io.StreamWriter(outdir, freq, pol)

    integrations = 0
    while integrations * TIME_PER_INTEGRATION < TOTAL_TIME:

        this_ntime = random.randint(16, 1024)

        # Time axis.
        this_time = np.arange(integrations, integrations + this_ntime,
                              dtype=np.float64) * TIME_PER_INTEGRATION
        # The data.
        this_intensity = np.empty((NFREQ, NPOL, this_ntime), dtype=np.float32)
        this_intensity[:] = Tsys
        this_noise = random.randn(*(this_intensity.shape))
        this_noise *= Tsys / np.sqrt(nsamp_integration)
        # Discretize.
        this_noise = np.round(this_noise)
        this_intensity += this_noise
        # Weights
        this_weight = np.empty((NFREQ, NPOL, this_ntime), dtype=np.uint8)
        # Full weight for now.
        this_weight[:] = 255

        writer.absorb_chunk(
                time=this_time,
                intensity=this_intensity,
                weight=this_weight,
                )

        integrations += this_ntime
    writer.finalize()



