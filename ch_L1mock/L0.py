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
from scipy.fftpack import fft, ifft
import ch_vdif_assembler

import io
import constants
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

    def __init__(self, nframe_integrate=512, **kwargs):
        super(BaseCorrelator, self).__init__(**kwargs)
        self._nframe_integrate = nframe_integrate

    @property
    def delta_t(self):
        return self.nframe_integrate / constants.FPGA_FRAME_RATE

    @property
    def nframe_integrate(self):
        return self._nframe_integrate

    def square_accumulate(self, efield, mask):
        return _L0.square_accumulate(efield, self._nframe_integrate)

    def process_chunk(self, t0, nt, efield, mask):
        ninteg = self._nframe_integrate
        if nt % ninteg:
            # This is currently true of all subclasses.
            msg = ("Number of samples to accumulate (%d) must evenly divide"
                   " number of samples (%d).")
            msg = msg % (ninteg, nt)
            raise ValueError(msg)

        #t0 = time.time()
        intensity, weight = self.square_accumulate(efield, mask)
        #print "Chunk integration time:", time.time() - t0

        time0 = float(t0) / self._nframe_integrate + 1. / 2
        time0 = time0 * self.delta_t

        self.post_process_intensity(time0, intensity, weight)

    def post_process_intensity(self, time0, intensity, weight):
        pass

    
class ReferenceSqAccumMixin(object):
    """Reference square accumulator, used for testing.

    This mixin can be used to replace the central engine of a correlator with a
    slow, reference, pure-python implementation. This can be usefull for
    testing.

    """

    byte_data = False

    def subband(fft, nchan=16,axis=2):
        # fft is an array of size constants.chime_nfreq, 2, nt
        # nt is a multiple of 2*nchan.
        # the output array is of shape nchan*constants.chime_nfreq, 2, nt/nchan

        nfreq = fft.shape[0]
        nt = fft.shape[2]
        assert nt%(2*nchan)==0
        chunksize = nt/2/nchan
        
        fft_subband = np.zeros([nchan*fft.shape[0],2,fft.shape[2]/nchan],dtype=fft.dtype)

        for i in range(nfreq):
            for j in range(nchan):
                fft_subband[i+nchan*j,:,0:chunksize] = fft[i,:,chunksize*j:chunksize*(j+1)]
                fft_subband[i+nchan*j,:,chunksize:2*chunksize] = fft[i,:,nt-chunksize*(j+1):nt-chunksize*(j)]
        return fft_subband

    def upchannelize(self, t0, nt, efield, mask, nchan):
        """ The 'efield' arg is a shape (nfreq,2,nt) complex array with electric field values, where
        the middle index is polarziation.  Missing data is represented by (0+0j).  The 'mask' arg
        is a shape (nfreq,2,nt) integer array which is 0 for missing data, and 1 for non-missing.
        This code currently does not handle the mask. I need to add how the mask is handled. Currently, it just assumes the data is really zero.
         """

        assert efield.shape == (constants.chime_nfreq, 2, nt)
        assert efield.dtype is np.complex64
        
        assert nt%(2*nchan)==0

        ninteg = self._nframe_integrate
        if nt % ninteg:
            # This is currently true of all subclasses.
            msg = ("Number of samples to accumulate (%d) must evenly divide"
                   " number of samples (%d).")
            msg = msg % (ninteg, nt)
            raise ValueError(msg)
        # But since we have increased the frequency sampling, the sampling time has increased by a factor of nchan. Thus the nframe_integrate must reduce by a factor of nchan. We need _nframe_integrate to be divisible by nchan.
        assert ninteg %nchan ==0

        # take fft of efield
        efield_fft = fft(efield,axis=2,overwrite_x=False)

        # heterodyne the array to get the sub bands
        # we want to split each band into 16 sub bands. 
        # To avoid interpolation, lets insist that the 
        # FFT length is an integer multiple of twice the 
        # number of sub bands.
        efield_sub_fft = subbband(efield_fft,nchan=16,axis=2)

        # invert fft
        efield_sub = ifft(efield_sub_fft,axis=2,overwrite_x=False)

        # square and integrate. mask is ignored anyway.
        intensity, weight = self.square_accumulate(efield, mask=np.ones_like(efield), self._nframe_integrate/nchan)

        frame = t0 // constants.timestamps_per_frame
        
        time0 = float(t0) / self._nframe_integrate + 1. / 2
        time0 = time0 * self.delta_t


    def square_accumulate(self, efield, mask, ninteg=None):

        # Moved ninteg to the argument to allow 
        if (ninteg is None) or (type(ninteg) is not int):
            ninteg = self._nframe_integrate

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
        self._finalizes = []

    def add_callback(self, callback, finalize=None):
        """Add post processing to the correlator.

        The argument `callback` must be a function with the call signature
        `callback(t0, intensity, weight)`.

        """

        self._callbacks.append(callback)
        if finalize is not None:
            self._finalizes.append(finalize)

    def add_diskwrite_callback(self, stream_writer):
        self._stream_writer = stream_writer
        def wrap_absorb(time0, intensity, weight):
            time = time0 + np.arange(intensity.shape[2]) * self.delta_t
            self._stream_writer.absorb_chunk(
                    time=time,
                    intensity=intensity,
                    weight=weight,
                    )
        self.add_callback(wrap_absorb, stream_writer.finalize)

    def post_process_intensity(self, time0, intensity, weight):
        for c in self._callbacks:
            c(time0, intensity, weight)

    def finalize(self):
        for c in self._finalizes:
            c()


class DiskWriteCorrelator(CallBackCorrelator):
    """Correlator that streams output to disk.

    """

    def __init__(self, *args, **kwargs):
        outdir = kwargs.pop('outdir', '')
        super(DiskWriteCorrelator, self).__init__(*args, **kwargs)
        stream_writer = io.StreamWriter(outdir)
        self.add_diskwrite_callback(stream_writer)





