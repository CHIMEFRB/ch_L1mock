"""Library of L1 mock preprocessing tasks.

Preprocessors modify the data and weights in place prior to dedispersion. For
more information see the bonsai header file, especially
dedisperser::preprocess_data.

Feel free to write your own and add them to the INDEX.

"""

import abc

import numpy as np


class BasePreprocessor(object):
    """Abstract base class for preprocessors.

    All preprocessors must inherit from this.  When subclassing, you may add
    only keyword arguments to __init__.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, dedisperser):
        self._dedisperser = dedisperser

    @property
    def dedisperser(self):
        return self._dedisperser

    @abc.abstractmethod
    def __call__(self, data, weights):
        pass


class BurstSearchInject(BasePreprocessor):
    """Inject simulated events using Burst Search package"""

    def __init__(self, dedisperser, **kwargs):
        super(BurstSearchInject, self).__init__(dedisperser)
        from burst_search import simulate

        delta_t = dedisperser.dt_sample
        freq = np.linspace(
                dedisperser.freq_hi_MHz,
                dedisperser.freq_lo_MHz,
                dedisperser.nfreq,
                endpoint=False,
                )
        # burst_search.simulation expects a burst_search style datasource. Just
        # fake it.
        class DummyDataSource(object):
            pass
        ds = DummyDataSource()
        ds.delta_t = delta_t
        ds.freq = freq
        self._simulator = simulate.EventSimulator(ds, **kwargs)
        self._chunk_ind = 0

    def __call__(self, data, weights):
        t0 = self._chunk_ind * data.shape[1] * self.dedisperser.dt_sample
        self._simulator.inject_events(t0, data)
        self._chunk_ind += 1


class BurstSearchDefault(BasePreprocessor):
    """Default preprocessing base on the Burst Search package."""

    def __init__(self, dedisperser):
        super(BurstSearchDefault, self).__init__(dedisperser)

    @staticmethod
    def subtract_weighted_time_mean(data, weights):
        mean, bad_freq = _weighted_time_mean(data, weights)
        data -= mean[:,None]
        data[bad_freq] = 0

    def __call__(self, data, weights):
        from burst_search import preprocess
        self.subtract_weighted_time_mean(data, weights)
        # The following mostly ignore the weights and don't update it.
        preprocess.remove_outliers(data, 5)
        self.subtract_weighted_time_mean(data, weights)
        preprocess.remove_bad_times(data, 3)
        preprocess.remove_noisy_freq(data, 3)


class ThermalNoiseWeight(BasePreprocessor):
    """Assuming input weights are normalized such that full weight is unity,
    rescale the weights such that data * weights has unit variance if the data
    is thermal noise dominated.

    """

    def __init__(self, dedisperser):
        super(ThermalNoiseWeight, self).__init__(dedisperser)

    def __call__(self, data, weights):
        dedisperser = self.dedisperser
        full_band_width = dedisperser.freq_hi_MHz - dedisperser.freq_lo_MHz
        full_band_width *= 1e6
        delta_f = full_band_width / data.shape[0]
        delta_t = dedisperser.dt_sample

        time_mean, bad_freq = _weighted_time_mean(data, weights)
        # Factor of 2 for polarizations.
        std_thermal = time_mean / np.sqrt(2 * delta_t * delta_f)
        std_thermal[bad_freq] = 1
        weights /= std_thermal[:, None]
        weights[bad_freq] = 0



# Index of preprocessing tasks used for lookup by manager.Manager.
INDEX = {
        'burst_search_inject' : BurstSearchInject,
        'burst_search_default' : BurstSearchDefault,
        'thermal_noise_weight' : ThermalNoiseWeight,
        }


# Helper functions.

def _weighted_time_mean(data, weights):
    num = np.sum(data * weights, -1)
    den = np.sum(weights, -1)
    bad_freq = den < 0.001 * np.mean(den)
    den[bad_freq] = 1
    mean = num / den
    mean[bad_freq] = 0
    return mean, bad_freq

