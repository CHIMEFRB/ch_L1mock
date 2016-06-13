import unittest
import Queue

import numpy as np
import ch_vdif_assembler

from ch_L1mock import datasource, L0, utils
from ch_L1mock.tests import data as testdata


class TestVdifSource(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=6., nsec=2)
        assembler = ch_vdif_assembler.assembler()
        p = DummyCorrelator(nframe_integrate=512)
        ntime_chunk = 563
        ds = datasource.vdifSource(p, ntime_chunk=ntime_chunk)
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
            self.assertEqual(data.shape, (1024, ntime_chunk))
            self.assertTrue(np.dtype(data.dtype) is np.dtype(np.float32))
            time = ds.last_time
            # See L0.DummyDataMixing for data expectation.
            expectation = freq[:,None] * time * 3
            self.assertTrue(np.allclose(data[mask==1], expectation[mask==1]))
            time_mask_inds = (time * 100) % 20 < 1
            self.assertTrue(np.all(np.sum(mask, 0)[time_mask_inds] == 0))
            freq_mask_inds = (freq / 1e6) % 40 < 1
            self.assertTrue(np.all(np.sum(mask, 1)[freq_mask_inds] == 0))

            n_data += data.shape[1]
        thread.check_join()
        ds.finalize()
        self.assertGreater(n_data, 1000)


class DummyCorrelator(L0.DummyDataMixin, L0.CallBackCorrelator):
    pass


class TestDiskSource(unittest.TestCase):

    def test_runs(self):
        ntime_chunk = 667
        ds = datasource.DiskSource(testdata.TEST_DATA_DIR,
                ntime_chunk=ntime_chunk)
        n_data = 0

        while True:
            try:
                data, mask = ds.yield_chunk()
            except StopIteration:
                break
            self.assertEqual(data.shape, (128, ntime_chunk))
            self.assertTrue(np.dtype(data.dtype) is np.dtype(np.float32))
            n_data += data.shape[1]
        ds.finalize()
        self.assertGreater(n_data,
                testdata.TOTAL_TIME / testdata.TIME_PER_INTEGRATION)



if __name__ == '__main__':
    unittest.main()

