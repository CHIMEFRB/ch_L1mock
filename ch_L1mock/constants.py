"""
CHIME constants and parameters.

"""

# Sampling frequency
ADC_SAMPLE_RATE = float(800e6)

# Number of samples in the inital FFT in the F-engine.
FPGA_NSAMP_FFT = 2048

FPGA_FRAME_RATE = ADC_SAMPLE_RATE / FPGA_NSAMP_FFT

# f-engine parameters for alias sampling.
FPGA_FREQ0 = ADC_SAMPLE_RATE
FPGA_NFREQ = FPGA_NSAMP_FFT / 2
FPGA_DELTA_FREQ = - ADC_SAMPLE_RATE / FPGA_NSAMP_FFT
