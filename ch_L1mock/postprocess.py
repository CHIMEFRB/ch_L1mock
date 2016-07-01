"""Library of L1 mock postprocessing tasks.

Postprocessing tasks are classes that:
    1) Can be initialized with one posiitonal arguemnt (dedisperser) only keyword arguments, and
    2) Are callable with the signature `task_instance(itree,
    trigger_set)`
    4) Return a list of Event objects (which may be empty).
    3) They *may* modify trigger_set in place.

Feel free to write your own and register them in the INDEX.

"""

import abc

import numpy as np



class Event(object):
    """Represents an L1 trigger.

    Should eventually contain all/most information that an L1 trigger packet
    would.

    """

    def __init__(self, dedisperser, itree, trigger_set, index):
        self._index = index
        self._snr = trigger_set.trigger_max[index]



class BasePostProcessor(object):
    """Abstract base class for post processors."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, dedisperser):
        self._dedisperser = dedisperser

    @property
    def dedisperser(self):
        return self._dedisperser

    @abc.abstractmethod
    def __call__(self, itree, trigger_set):
        return []



class SimpleThreshold(BasePostProcessor):

    def __init__(self, dedisperser, threshold=10.):
        super(SimpleThreshold, self).__init__(dedisperser)
        self._threshold = threshold

    def __call__(self, itree, trigger_set):
        events = super(SimpleThreshold, self).__call__(itree, trigger_set)

        event_inds = np.argwhere(trigger_set.trigger_max > self._threshold)

        for event_ind in event_inds:
            events.append(Event(self.dedisperser, itree, trigger_set,
                tuple(event_ind)))
        return events



INDEX = {
        'simple_threshold' : SimpleThreshold,
        }



