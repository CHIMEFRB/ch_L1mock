"""
Data source interface for the L1 mock up.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import Queue

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
        """String identifying the source stream

        This is used for plot labeling, etc. The output should be appropriate
        for use within file names (avoid spaces and a few other characters).

        """
        pass



    # These methods are not strictly abstract, in that they do useful work and
    # may/should be called in subclass implementations using `super`.

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
        mask = np.zeros((self.nfreq, self.ntime), dtype=np.uint8)
        return data, mask


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



class vdifSource(DataSource):

    def __init__(self, processor, ntime_chunk):
        super(vdifSource, self).__init__()
        self._ntime_chunk = ntime_chunk
        self._processor = processor
        self._correlated_data_queue = Queue.Queue()
        processor.add_callback(self._absorb_chunk, self._end_stream)

    def __str__(self):
        return "vdif_data"

    @property
    def time0(self):
        # This is unknown and I don't know how to figure it out.
        return 0

    @property
    def ntime_chunk(self):
        return self._ntime_chunk

    @property
    def delta_t(self):
        return self._processor.delta_t

    @property
    def freq0(self):
        return self._processor.freq0

    @property
    def nfreq(self):
        return self._processor.nfreq

    @property
    def delta_f(self):
        return self._processor.delta_f


    def _absorb_chunk(self, time0, intensity, weight):
        print time0, intensity.shape
        self._correlated_data_queue.put(None)

    def _end_stream(self):
        self._correlated_data_queue.put(None)

    def yield_chunk(self):
        print "waiting for a chunk"
        data_and_mask = self._correlated_data_queue.get()
        if data_and_mask is None:
            raise StopIteration()
        else:
            self._current_chunk += 1
            return data_and_mask

