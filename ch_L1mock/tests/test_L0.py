import unittest

import numpy as np
import ch_vdif_assembler

from ch_L1mock import L0, constants


class TestReferenceIntegrator(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(gbps=0.1, nsec=2)
        assembler = ch_vdif_assembler.assembler()
        class ReferenceCorrelator(L0.ReferenceSqAccumMixin,
                L0.CallBackCorrelator):
            pass
        p1 = ReferenceCorrelator(nframe_integrate=512)
        p2 = L0.CallBackCorrelator(nframe_integrate=512)
        comparison = IntegratorComparison(p1, p2)
        assembler.register_processor(p1)
        assembler.register_processor(p2)
        assembler.run(stream)

    def test_fast_enough(self):
        stream = ch_vdif_assembler.make_simulated_stream(nsec=5)
        assembler = ch_vdif_assembler.assembler()
        p = L0.BaseCorrelator(nframe_integrate=512)
        assembler.register_processor(p)
        assembler.run(stream)

class TestToDisk(unittest.TestCase):

    def test_runs(self):
        stream = ch_vdif_assembler.make_simulated_stream(nsec=5)
        assembler = ch_vdif_assembler.assembler()
        p = L0.DiskWriteCorrelator(nframe_integrate=512, outdir='tmp_corr')
        assembler.register_processor(p)
        assembler.run(stream)


class TestUpChannelizing(unittest.TestCase):

    def test__ref_upchannelize_zero_input(self):
        """ Tests zero input return from _ref_upchannelize    """
        zero_efield = np.zeros([constants.FPGA_NFREQ,2,1024],dtype=np.complex64)
        mask = np.ones_like(zero_efield,dtype=np.uint8)

        efield_sub, mask_sub = L0._ref_upchannelize(zero_efield,mask,16)
        # check return shape
        self.assertEquals(efield_sub.shape,(constants.FPGA_NFREQ*16,2,1024/16))
        # check return values
        self.assertTrue(np.all(efield_sub == np.zeros([
            constants.FPGA_NFREQ*16,2,1024/16],dtype=np.complex64)))
        self.assertTrue(np.all(mask_sub == np.ones([
            constants.FPGA_NFREQ*16,2,1024/16],dtype=np.uint8)))
        
    def test__ref_upchannelize(self):
        """ Tests the fourier transform functionality"""
        # we create data from an easy but distinct fft form.
        # we will use 32 sample fft.
        efield_fft = np.zeros([1024,2,32],dtype=np.complex64)
        efield_fft[0,0,::2] = np.arange(16)
        
        efield = L0.ifft(efield_fft,axis=2,overwrite_x=False)

        # create the correct answer for nchan=16.
        efield_sub_fft = np.zeros([1024*16,2,2],dtype=np.complex64)
        efield_sub_fft[:16:2,0,0] = np.arange(8)
        efield_sub_fft[1:16:2,0,1] = np.flipud(np.arange(8,16))
        
        # Call upchannelize with None mask
        efield_sub, mask = L0._ref_upchannelize(efield,None,16)
        self.assertTrue(np.allclose(L0.fft(efield_sub),efield_sub_fft,atol=1E-6))        

    def test__subband_mixing(self):
        """ Tests the shuffling in _subband """
        data = np.zeros([1,1,32],dtype=np.uint8)
        data[0,0,:] = np.arange(32)
        chan_16_data = np.zeros([16,1,2],dtype=np.uint8)
        chan_16_data[:,0,0] = np.arange(16)
        chan_16_data[::-1,0,1] = np.arange(16,32)

        self.assertTrue(np.all(L0._subband(data,nchan=16,axis=2) == chan_16_data))
        
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
