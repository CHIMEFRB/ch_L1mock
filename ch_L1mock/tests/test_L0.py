import unittest

import ch_vdif_assembler

from ch_L1mock import L0


class TestReferenceIntegrator(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=1., nsec=5)
        assembler = ch_vdif_assembler.assembler()
        p = L0.ReferenceIntegrator(nsamp_integrate=512)
        assembler.register_processor(p)
        assembler.run(stream)



if __name__ == '__main__':
    unittest.main()
