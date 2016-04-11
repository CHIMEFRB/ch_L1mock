"""
Faster L0 routines.

"""

from libc.stdint cimport uint8_t

import numpy as np

cimport numpy as np
cimport cython


np.import_array()


cdef extern from "ch_vdif_assembler_kernels.hpp" namespace "ch_vdif_assembler":
    void _sum16_auto_correlations(int &sum, int &count, uint8_t *buf)


#@cython.boundscheck(False)
#@cython.wraparound(False)
def square_accumulate(
        np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] byte_data,
        int nsamp_integrate,
        ):

        cdef np.ndarray[dtype=np.float32_t, ndim=3, mode="c"] intensity,
        cdef np.ndarray[dtype=np.uint8_t, ndim=3, mode="c"] weight,

        shape = byte_data.shape

        cdef int ntime = shape[2]
        cdef int ntime_out = ntime / nsamp_integrate

        if ntime % nsamp_integrate:
            msg = ("Number of samples to integrate (%d) must evenly divide"
                   "number of samples (%d).")
            msg = msg % (nsamp_integrate, ntime)
            raise ValueError(msg)

        out_shape = (shape[0], shape[1], ntime_out)

        intensity = np.empty(out_shape, dtype=np.float32)
        weight = np.empty(out_shape, dtype=np.uint8)

        # Do stuff

        return intensity, weight
