import unittest

import ch_vdif_assembler

from ch_L1mock import L0


class TestReferenceIntegrator(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=0.1, nsec=2)
        assembler = ch_vdif_assembler.assembler()
        p1 = L0.ReferenceIntegrator(nsamp_integrate=512)
        p2 = L0.FastIntegrator(nsamp_integrate=512)
        comparison = L0.IntegratorComparison(p1, p2)
        assembler.register_processor(p1)
        assembler.register_processor(p2)
        assembler.run(stream)

    def test_fast_enough(self):
        stream = ch_vdif_assembler.make_simulated_stream(nsec=5)
        assembler = ch_vdif_assembler.assembler()
        p = L0.FastIntegrator(nsamp_integrate=512)
        assembler.register_processor(p)
        assembler.run(stream)


if __name__ == '__main__':
    unittest.main()
