import unittest

import numpy as np
import ch_vdif_assembler

from ch_L1mock import L0


class TestReferenceIntegrator(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=0.1, nsec=2)
        assembler = ch_vdif_assembler.assembler()
        class ReferenceCorrelator(L0.ReferenceSqAccumMixin,
                L0.CallBackCorrelator):
            pass
        p1 = ReferenceCorrelator(nsamp_integrate=512)
        p2 = L0.CallBackCorrelator(nsamp_integrate=512)
        comparison = IntegratorComparison(p1, p2)
        assembler.register_processor(p1)
        assembler.register_processor(p2)
        assembler.run(stream)

    def test_fast_enough(self):
        stream = ch_vdif_assembler.make_simulated_stream(nsec=5)
        assembler = ch_vdif_assembler.assembler()
        p = L0.BaseCorrelator(nsamp_integrate=512)
        assembler.register_processor(p)
        assembler.run(stream)



# Testing Classes
# ===============

class IntegratorComparison(object):

    integrated_chunk1 = None
    integrated_chunk2 = None

    def __init__(self, integrator1, integrator2):
        integrator1.add_callback(self.add_integrated_chunk1)
        integrator2.add_callback(self.add_integrated_chunk2)

    def add_integrated_chunk1(self, t0, intensity, weight):
        self.integrated_chunk1 = (t0, intensity, weight)
        if self.integrated_chunk2:
            self.compare()

    def add_integrated_chunk2(self, t0, intensity, weight):
        self.integrated_chunk2 = (t0, intensity, weight)
        if self.integrated_chunk1:
            self.compare()

    def compare(self):
        c1 = self.integrated_chunk1
        c2 = self.integrated_chunk2
        if not np.allclose(c1[0], c2[0]):
            raise RuntimeError("Time stamps don't match")
        if not np.allclose(c1[1], c2[1]):
            raise RuntimeError("Intensity does not match")
        if not np.allclose(c1[2], c2[2]):
            raise RuntimeError("Weight does not match")
        self.integrated_chunk1 = None
        self.integrated_chunk2 = None
        print "Passed!"

if __name__ == '__main__':
    unittest.main()
