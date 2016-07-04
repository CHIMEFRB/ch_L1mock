"""Library of L1 mock post processing tasks.

Post processing tasks work on the dedispersed, coarse-grained data.
They may modify this data in place, return a list of ``Event`` objects
or both.  For more information see the Bonsai header file, especially
dedisperser::process_triggers.

Feel free to write your own task and add it to INDEX.

"""

import abc

import numpy as np



class Event(object):
    """Represents an L1 event.

    Should eventually contain all/most information that an L1 output packet
    would.

    """

    def __init__(self, dedisperser, itree, trigger_set, index):
        self._index = index
        self._snr = trigger_set.trigger_max[index]


class BasePostProcessor(object):
    """Abstract base class for post processors.

    All post processors must inherit from this class.

    When sub-classing, only keyword arguments may be added to the constructor.

    """

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
    """Anything over the threshold."""

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



