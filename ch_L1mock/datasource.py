"""
Data source interface for the L1 mock up.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from os import path
import Queue

import numpy as np

from ch_frb_io import stream as io


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
    def first_time0(self):
        """Start time of the data stream.

        Must return a time, in seconds, refering to the first sample of the
        first chunk.

        Not guaranteed to be present until after the first chunk is yielded.

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

    def yield_chunk(self, timeout=None):
        """Yield a chunk of data.

        Base class implementation just initializes the data array and keeps
        track of the chuck number.

        Returns
        -------
        data : numpy array with shape (nfreq, ntime) and dtype np.float32.

        Raises
        ------
        NoData : If a blocking oberation times out.

        """

        self._current_chunk += 1
        data = np.empty((self.nfreq, self.ntime_chunk), dtype=np.float32)
        mask = np.zeros((self.nfreq, self.ntime_chunk), dtype=np.uint8)
        return data, mask

    def finalize(self):
        pass


    # These are mixin methods and do not need to be overridden.

    @property
    def current_chunk(self):
        """Number of chunks yielded so far.

        This is zero on initialization and increments for each call to
        :meth:`self.yield_chunk`

        """

        return self._current_chunk

    @property
    def last_time0_offset(self):
        return self.delta_t * self.ntime_chunk * (self.current_chunk - 1)

    @property
    def last_time0(self):
        """Start time of the most recent chunk.

        Time, in seconds, of the centre of the first sample of the most
        recently yielded chunk.

        """

        return self.first_time0 + self.last_time0_offset

    @property
    def last_time(self):
        return (self.last_time0 + np.arange(self.ntime_chunk, dtype=np.float64)
                * self.delta_t)

    @property
    def freq(self):
        return (self.freq0 + np.arange(self.nfreq, dtype=np.float64)
                * self.delta_f)



class NoData(Exception):
    pass


class vdifSource(DataSource):

    def __init__(self, processor, ntime_chunk):
        super(vdifSource, self).__init__()
        self._ntime_chunk = ntime_chunk
        self._processor = processor
        processor.add_callback(self._absorb_chunk, self._end_stream)

        self._correlated_data_queue = Queue.Queue()
        self._first_time0 = None
        self._initialize_out_chunk()
        self._total_samples = 0
        self._first_time0 = None

    def __str__(self):
        return "vdif"

    @property
    def first_time0(self):
        # This is unknown and I don't know how to figure it out.
        return self._first_time0

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

    def _initialize_out_chunk(self):
        self._out_chunk = np.empty((self.nfreq, self.ntime_chunk),
                                   dtype=np.float32)
        self._out_mask = np.empty((self.nfreq, self.ntime_chunk),
                                  dtype=np.uint8)

    def _absorb_chunk(self, time0, intensity, weight):
        # If this is the first chunk, initialize the stream start time.
        if self._first_time0 is None:
            self._first_time0 = time0

        total_samples = self._total_samples
        ntime_in = intensity.shape[2]
        ntime_out = self._ntime_chunk
        start_ind = int(round((time0 - self._first_time0) / self.delta_t))

        # Process the data.
        data, mask = _format_intensity_weight(intensity, weight)

        # Fill in and "gap" indecies with zeros.
        if start_ind > total_samples:
            out_edges = range((total_samples // ntime_out + 1) * ntime_out,
                              start_ind, ntime_out)
            out_edges = [total_samples] + out_edges + [start_ind]
            #print "gap:", out_edges
            for ii in range(len(out_edges) - 1):
                left_edge = out_edges[ii] % ntime_out
                right_edge = (out_edges[ii + 1] - 1) % ntime_out + 1
                self._out_chunk[:, left_edge:right_edge] = 0
                self._out_mask[:, left_edge:right_edge] = 0
                if right_edge == ntime_out:
                    self._correlated_data_queue.put((
                        self._out_chunk, self._out_mask))
                    self._initialize_out_chunk()

        # Now deal with samples in the chunk.
        out_edges = range((start_ind // ntime_out + 1) * ntime_out,
                              start_ind + ntime_in, ntime_out)
        out_edges = [start_ind] + out_edges + [start_ind + ntime_in]
        in_edges = [ind - start_ind for ind in out_edges]
        for ii in range(len(out_edges) - 1):
            left_edge_out = out_edges[ii] % ntime_out
            right_edge_out = (out_edges[ii + 1] - 1) % ntime_out + 1
            left_edge_in = in_edges[ii]
            right_edge_in = in_edges[ii + 1]
            self._out_chunk[:, left_edge_out:right_edge_out] = \
                    data[:, left_edge_in:right_edge_in]
            self._out_mask[:, left_edge_out:right_edge_out] = \
                    mask[:, left_edge_in:right_edge_in]
            if right_edge_out == ntime_out:
                self._correlated_data_queue.put((
                    self._out_chunk, self._out_mask))
                self._initialize_out_chunk()

        self._total_samples = start_ind + ntime_in


    def _end_stream(self):
        self._correlated_data_queue.put(None)

    def yield_chunk(self, timeout=None):
        try:
            data_and_mask = self._correlated_data_queue.get(timeout=timeout)
        except Queue.Empty:
            raise NoData
        if data_and_mask is None:
            raise StopIteration()
        else:
            self._current_chunk += 1
            return data_and_mask


class DiskSource(DataSource):

    def __init__(self, datadir, ntime_chunk):
        super(DataSource, self).__init__()
        self._datadir = datadir
        self._ntime_chunk = ntime_chunk
        self._current_chunk = 0

        self._stream = io.StreamReader(datadir)
        freq = self._stream.freq
        delta_f = np.mean(np.diff(freq))
        if not np.allclose(np.diff(freq), delta_f):
            raise ValueError("Frequencies not uniformly spaced")
        self._delta_f = delta_f
        self._freq0 = freq[0]
        self._nfreq = len(freq)
        self._delta_t = np.median(np.diff(self._stream.time))
        self._first_time0 = self._stream.time[0]

    def __str__(self):
        dir_base_name = path.basename(path.abspath(self._datadir))
        return "disk_" + dir_base_name

    @property
    def freq0(self):
        return self._freq0

    @property
    def delta_f(self):
        return self._delta_f

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def first_time0(self):
        return self._first_time0

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def ntime_chunk(self):
        return self._ntime_chunk

    def yield_chunk(self, timeout=None):
        start_of_chunk = self.last_time0 + self.delta_t * self.ntime_chunk
        end_of_chunk = self.last_time0 + self.delta_t * self.ntime_chunk * 2
        last_ind = np.searchsorted(self._stream.time, end_of_chunk -
                self.delta_t/10)
        ntime_read = last_ind - self._stream.current_time_ind

        datasets = self._stream.yield_chunk(ntime_read)
        time = datasets['time']
        intensity = datasets['intensity']
        weight = datasets['weight']
        data, mask = _format_intensity_weight(intensity, weight)

        # Deal with lost chunks and data gaps!
        if data.shape[-1] != self.ntime_chunk:
            new_shape = data.shape[:-1] + (self.ntime_chunk,)
            data_new = np.zeros(new_shape, dtype=data.dtype)
            mask_new = np.zeros(new_shape, dtype=mask.dtype)
            inds = np.round((time - start_of_chunk) / self.delta_t).astype(int)
            # This is probably very slow. Should try to convert to slice where
            # possible.
            data_new[...,inds] = data
            mask_new[...,inds] = mask
            data = data_new
            mask = mask_new
        elif not np.allclose(np.diff(time), self.delta_t):
            raise ValueError("Non uniform time axis.")

        self._current_chunk += 1
        return data, mask


def _format_intensity_weight(intensity, weight):
    in_shape = intensity.shape
    data = np.empty((in_shape[0], in_shape[2]), dtype=np.float32)
    data[:,:] = intensity[:,0,:]
    data += intensity[:,1,:]
    mask = np.ones(data.shape, dtype=np.uint8)
    # Mask data with less than 50% weight.
    mask[np.logical_or(weight[:,0,:] < 128, weight[:,1,:] < 128)] = 0
    return data, mask
