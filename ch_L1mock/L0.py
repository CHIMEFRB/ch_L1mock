"""
Implementation of L0 processing on for pathfinder pulsar beams.

L0 includes all correlator side processing, including beam forming,
up-channelization, square accumulation, and initial dedispersion. In the
pathfinder mock up, data is recieved as beam formed baseband data. This module
is based off of the packet assembly package: 'ch_vdif_assembler'.

"""

import logging
import time

import numpy as np
import ch_vdif_assembler

import io
import _L0


logger = logging.getLogger(__name__)


# Vdif processors
# ===============

class BaseCorrelator(ch_vdif_assembler.processor):
    """Abstract base class for correlators.

    Subclasses should implement `post_process_intensity` to do something with
    correlated data.

    """

    byte_data = True

    def __init__(self, nsamp_integrate=512, **kwargs):
        super(BaseCorrelator, self).__init__(**kwargs)
        self._nsamp_integrate = nsamp_integrate

    def square_accumulate(self, efield, mask):
        return _L0.square_accumulate(efield, self._nsamp_integrate)

    def process_chunk(self, t0, nt, efield, mask):
        ninteg = self._nsamp_integrate
        if nt % ninteg:
            # This is currently true of all subclasses.
            msg = ("Number of samples to accumulate (%d) must evenly divide"
                   " number of samples (%d).")
            msg = msg % (ninteg, nt)
            raise ValueError(msg)

        #t0 = time.time()
        intensity, weight = self.square_accumulate(efield, mask)
        #print "Chunk integration time:", time.time() - t0
        self.post_process_intensity(t0, intensity, weight)

    def post_process_intensity(self, t0, intensity, weight):
        pass


class ReferenceSqAccumMixin(object):
    """Reference square accumulator, used for testing.

    This mixin can be used to replace the central enging of a correlator with a
    slow, reference, pure-python implementation. This can be usefull for
    testing.

    """

    byte_data = False

    def square_accumulate(self, efield, mask):
        ninteg = self._nsamp_integrate

        e_squared = abs(efield)**2
        shape = efield.shape
        new_shape = shape[:-1] + (shape[-1] // ninteg, ninteg)
        e_squared.shape  = new_shape
        mask.shape = new_shape

        # Integrate.
        intensity = np.sum(e_squared, -1, dtype=np.float32)
        weight = np.sum(mask, -1, dtype=np.float32)
        # Normalize for missing data.
        bad_inds = weight == 0
        weight[bad_inds] = 1
        intensity *= ninteg / weight
        # Convert weight to integer between 0 and 255.
        weight *= 255 / ninteg
        weight = np.round(weight).astype(np.uint8)
        weight[bad_inds] = 0

        return intensity, weight


class CallBackCorrelator(BaseCorrelator):
    """Correlator to which post processing can be added dynamically.

    """

    def __init__(self, *args, **kwargs):
        super(CallBackCorrelator, self).__init__(*args, **kwargs)
        self._callbacks = []

    def add_callback(self, callback):
        """Add post processing to the correlator.

        The argument `callback` must be a function with the call signature
        `callback(t0, intensity, weight)`.

        """

        self._callbacks.append(callback)

    def post_process_intensity(self, t0, intensity, weight):
        for c in self._callbacks:
            c(t0, intensity, weight)


class DiskWriteCorrelator(BaseCorrelator):
    """Correlator that streams output to disk.

    """

    def __init__(self, *args, **kwargs):
        outdir = kwargs.pop('outdir', '')
        super(DiskWriteCorrelator, self).__init__(*args, **kwargs)
        pol = ['XX', 'YY']    # XXX Wrong
        freq = np.arange(1024)    # XXX Wrong
        self._stream_writer = io.StreamWriter(outdir, freq, pol)

    def post_process_intensity(self, t0, intensity, weight):
        time = np.arange(intensity.shape[2])    # XXX Wrong. Placeholder.
        self._stream_writer.absorb_chunk(
                time=time,
                intensity=intensity,
                weight=weight)

    def finalize(self):
        self._stream_writer.finalize()




