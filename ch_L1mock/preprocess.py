"""Library of L1 mock preprocessing tasks.

Preprocessing tasks are classes that:
    1) Can be initialized with only keyword arguments, and
    2) Are callable with the signature `task_instance(dedisperser, data, weights)` and
    returns `None`.

Feel free to write your own and register them in the INDEX.

"""

import numpy as np



class BurstSearchDefault(object):

    def __init__(self):
        pass

    @staticmethod
    def subtract_weighted_time_mean(data, weights):
        num = np.sum(data * weights, -1)
        den = np.sum(weights, -1)
        bad_freq = den < 0.001 * np.mean(den)
        den[bad_freq] = 1
        data -= (num / den)[:,None]
        data[bad_freq] = 0

    def __call__(self, dedisperser, data, weights):
        from burst_search import preprocess
        self.subtract_weighted_time_mean(data, weights)
        # The following mostly ignore the mask and don't update it.
        preprocess.remove_outliers(data, 5)
        self.subtract_weighted_time_mean(data, weights)
        preprocess.remove_bad_times(data, 3)
        preprocess.remove_noisy_freq(data, 3)






# Index of preprocessing tasks used for lookup by manager.Manager.
INDEX = {
        'burst_search_default' : BurstSearchDefault,
        }
