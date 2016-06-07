"""
Data source interface for the L1 mock up.
"""

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np


class DataSource(object):
    """Abstract base class for data sources.

    Data sources both provide the data and describe the data.

    Note that data must be contiguous in time, both within a chunk and from
    chunk to chunk.

    """

    __metaclass__ = ABCMeta

    # The following methods and properties are abstract. They *must* be
    # implemented by the subclass.

    @abstractproperty
    def freq0(self):
        pass

    @abstractproperty
    def nfreq(self):
        pass

    @abstractproperty
    def delta_f(self):
        """This may be negative (in fact it usually is)."""
        pass

    @abstractproperty
    def time0(self):
        """Start time of the data stream.

        Must return the Unix/Posix time of the centre of the first sample of
        the first chunk.

        """
        pass

    @abstractproperty
    def ntime_chunk(self):
        pass

    @abstractproperty
    def delta_t(self):
        pass

    @abstractmethod
    def __str__(self):
        """Used to identify the source stream, for plot labeling, etc."""



    # These methods are not strictly abstract, in that they do useful work and
    # may be called in subclass implementations using `super`.

    def __init__(self):
        self._current_chunk = 0

    def yield_chunk(self):
        """Yield a chunk of data.

        Base class implementation just initializes the data array and keeps
        track of the chuck number.

        Returns
        -------
        data : numpy array with shape (nfreq, ntime) and dtype np.float32.

        """

        self._current_chunk += 1
        data = np.empty((self.nfreq, self.ntime), dtype=np.float32)
        return data


    # These are mixin methods and do not need to be overridden.

    @property
    def current_chunk(self):
        """Number of chunks yeilded so far.

        This is zero on initialization and increments for each call to
        :meth:`self.yield_chunk`

        """

        return self._current_chunk

    @property
    def current_time_offset(self):
        return self.delta_t * self.ntime_chunk * self.current_chunk

    @property
    def current_time(self):
        """Start time of next chunk.

        Unix/Posix time of the centre of the first sample of the next chunk.

        """

        return self.time0 + self.current_time_offset




