"""
Faster L0 routines.

"""

from libc.stdint cimport uint8_t
from libc cimport math

import numpy as np

cimport numpy as np
cimport cython


np.import_array()

#do not square/sum here!
#cdef extern from "ch_vdif_assembler_kernels.hpp" namespace "ch_vdif_assembler":
#    void _sum16_auto_correlations(int &sum, int &count, uint8_t *buf) nogil

def convert_to_python(np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] byte_data not None,
    np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] mask not None):
        return byte_data.astype(np.float32), mask

@cython.boundscheck(False)
@cython.wraparound(False)
def square_accumulate(
        np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] byte_data not None,
        int nsamp_integrate,
        ):

        cdef np.ndarray[dtype=np.float32_t, ndim=3, mode="c"] intensity,
        cdef np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] weight,

        shape = byte_data.shape

        cdef int ntime = shape[2]
        cdef int ntime_out = ntime // nsamp_integrate

        if ntime % nsamp_integrate:
            msg = ("Number of samples to integrate (%d) must evenly divide"
                   "number of samples (%d).")
            msg = msg % (nsamp_integrate, ntime)
            raise ValueError(msg)
        if nsamp_integrate % 16:
            msg = ("Number of samples to accumulate (%d) must evenly divide"
                   " by 16.") % nsamp_integrate
            raise ValueError(msg)

        out_shape = (shape[0], shape[1], ntime_out)

        intensity = np.empty(out_shape, dtype=np.float32)
        weight = np.empty(out_shape, dtype=np.uint8)

        cdef int ii, jj, kk, hh
        cdef np.float32_t tmp_inten, tmp_weight
        cdef int sum, count, t_offset

        with nogil:
            for ii in range(shape[0]):
                for jj in range(shape[1]):
                    for kk in range(ntime_out):
                        sum = 0
                        count = 0
                        t_offset = kk * nsamp_integrate
                        for hh in range(0, nsamp_integrate, 16):
                            #_sum16_auto_correlations(
                            #        sum,
                            #        count,
                            #        &byte_data[ii,jj,t_offset + hh],
                            #        )
                            pass
                        if count == 0:
                            tmp_inten = 0
                            tmp_weight = 0
                        else:
                            tmp_inten = sum
                            tmp_inten = tmp_inten / count * nsamp_integrate
                            tmp_weight = count
                            tmp_weight = tmp_weight / nsamp_integrate * 255
                        intensity[ii,jj,kk] = tmp_inten
                        weight[ii,jj,kk] = <uint8_t> math.round(tmp_weight)

        return intensity, weight
