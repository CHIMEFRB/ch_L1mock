"""
Implementation of L0 processing on for pathfinder pulsar beams.

L0 includes all correlator side processing, including beam forming,
up-channelization, square accumulation, and initial dedispersion. In the
pathfinder mock up, data is recieved as beam formed baseband data. This module
is based off of the packet assembly package: 'ch_vdif_assembler'.

"""

import logging

import numpy as np
import ch_vdif_assembler


logger = logging.getLogger(__name__)


# Vdif processors
# ===============


class ReferenceIntegrator(ch_vdif_assembler.processor):

    def __init__(self, nsamp_integrate=512, **kwargs):
        super(ReferenceIntegrator, self).__init__(**kwargs)
        self._nsamp_integrate = nsamp_integrate

    def process_chunk(self, t0, nt, efield, mask):
        ninteg = self._nsamp_integrate
        if nt % ninteg:
            msg = ("nsamp_integrate (%d) must evenly divide number of"
                   " samples (%d).")
            msg = msg % (ninteg, nt)
            raise ValueError(msg)
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


