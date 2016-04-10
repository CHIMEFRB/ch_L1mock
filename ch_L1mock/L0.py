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
        super(self, ReferenceIntegrator).__init__(**kwargs)
        self._nsamp_integrate = nsamp_integrate

