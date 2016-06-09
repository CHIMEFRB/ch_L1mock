import unittest
import Queue

import numpy as np
import ch_vdif_assembler

from ch_L1mock import datasource, L0, utils


class TestVdifSource(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=1., nsec=10)
        assembler = ch_vdif_assembler.assembler()
        p = TestCorrelator(nframe_integrate=512)
        ds = datasource.vdifSource(p, ntime_chunk=256)
        freq = ds.freq
        assembler.register_processor(p)
        thread = utils.ExceptThread(
                target=assembler.run,
                args=(stream,),
                )
        thread.start()
        n_data = 0
        while True:
            thread.check()
            try:

                data, mask = ds.yield_chunk(timeout=0.1)
            except StopIteration:
                break
            except datasource.NoData:
                continue
            self.assertEqual(data.shape, (1024, 256))
            time = ds.last_time
            expectation = freq[:,None] * time * 3
            self.assertTrue(np.allclose(data, expectation))
            n_data += data.shape[1]
        thread.check_join()
        self.assertGreater(n_data, 1000)


class TestCorrelator(L0.DummyDataMixin, L0.CallBackCorrelator):
    pass


if __name__ == '__main__':
    unittest.main()



