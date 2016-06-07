import unittest

import numpy as np
import ch_vdif_assembler

from ch_L1mock import datasource, L0, utils


class TestVdifSource(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=1., nsec=10)
        assembler = ch_vdif_assembler.assembler()
        p = L0.CallBackCorrelator(nframe_integrate=512)
        ds = datasource.vdifSource(p, ntime_chunk=256)
        assembler.register_processor(p)
        thread, bucket = utils.start_daemon_thread(
                assembler.run,
                args=(stream,),
                )
        n_data = 0
        while True:
            try:
                data = ds.yield_chunk()
                self.assertEqual(data.shape, (1024, 256))
                n_data += data.shape[1]
            except StopIteration:
                break
        thread.join()
        self.assertGreater(n_data, 1000)


if __name__ == '__main__':
    unittest.main()



