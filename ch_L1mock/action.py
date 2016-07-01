"""Library of L1 mock action tasks.

Postprocessing tasks are classes that:
    1) Can be initialized with only keyword arguments, and
    2) Are callable with the signature `task_instance(dediserser, itree,
    event_list)`. *event_list* is a list of postprocess.Event objects.
    3) Return None.

Feel free to write your own and register them in the INDEX.

"""

import abc


class BaseAction(object):
    """Abstract base class for event actions."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, dedisperser):
        self._dedisperser = dedisperser

    @property
    def dedisperser(self):
        return self._dedisperser

    @abc.abstractmethod
    def __call__(self, itree, events):
        pass


class Print(object):

    def __init__(self, dediserser):
        pass

    def __call__(self, itree, events):
        for e in events:
            print e._index, e._snr

INDEX = {
        'print' : Print,
        }

