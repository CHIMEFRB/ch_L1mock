"""Library of L1 mock action tasks.

Action tasks to be performed on L1 events.

For more information see the Bonsai header file, especially
dedisperser::process_triggers.

Feel free to write your own and add them to the INDEX.

"""

import abc


class BaseAction(object):
    """Abstract base class for event actions.

    All action tasks must inherit from this class.

    When subclassing, only keyword arguments may be added to the constructor.

    """

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
    """Just print to std out."""

    def __init__(self, dediserser):
        pass

    def __call__(self, itree, events):
        for e in events:
            print e._index, e._snr


INDEX = {
        'print' : Print,
        }

