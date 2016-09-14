"""
Back end FRB search engine using the Burst Search package.

"""

from Queue import Queue

import numpy as np
from burst_search import datasource, manager, preprocess
from ch_frb_io import stream as io

import constants

PARAMETER_DEFAULTS = {
        'time_block' : 100,    # Seconds.
        'overlap' : 10,
        #'time_block' : 3,    # Seconds.
        #'overlap' : 1,
        'max_dm' : 500.,
        }



class DataSource(datasource.DataSource):

    def __init__(self, vdif_cb_processor, **kwargs):
        super(DataSource, self).__init__(
                source="vdif_assembler",
                **kwargs
                )

        vdif_cb_processor.add_callback(self.absorb_chunk, self.end_stream)

        p = vdif_cb_processor
        self._delta_t_native = p.delta_t
        self._nfreq = constants.FPGA_NFREQ
        self._freq0 = constants.FPGA_FREQ0 / 1e6
        self._delta_f = constants.FPGA_DELTA_FREQ / 1e6

        # Don't know these.
        self._mjd = 0.
        self._start_time = 0.

        # Max size blocks vdiff assembler if search falls behind.
        self._correlated_data_queue = Queue(maxsize=10)
        # Everything from here on is only used by the vdif threads.
        self._nblocks_fetched = 0

        # Initialize the buffers
        gran = 512   # pretty arbitrary.
        ntime_block = self._block / self._delta_t_native
        ntime_block = int(np.ceil(ntime_block / gran) * gran)
        ntime_overlap = self._overlap / self._delta_t_native
        ntime_overlap = int(np.ceil(ntime_overlap / gran) * gran)
        self._ntime_block = ntime_block
        self._ntime_overlap = ntime_overlap

        self._this_buf = np.zeros((self.nfreq, ntime_block), dtype=np.float32)
        self._next_buf = np.zeros((self.nfreq, ntime_block), dtype=np.float32)
        self._ntime_absorbed = 0
        self._ntime_this_buf = 0

    @property
    def nblocks_left(self):
        return self._correlated_data_queue.qsize()

    @property
    def nblocks_fetched(self):
        return self._nblocks_fetched

    def get_next_block_native(self):
        t0, data = self._correlated_data_queue.get()
        if data is None:
            raise StopIteration()
        self._nblocks_fetched += 1
        return t0, data

    def absorb_chunk(self, time0, intensity, weight):
        t0, intensity, weight = preprocess_chunk(time0, intensity, weight)
        if self._ntime_absorbed == 0:
            self._this_buf_t0 = t0
        this_ntime = intensity.shape[-1]

        start_ind = int(round((t0 - self._this_buf_t0) / self._delta_t_native))

        if start_ind + this_ntime > self._ntime_block - self._ntime_overlap:
            # XXX Assume whole chunk is in due to granularity.
            nb_start_ind = start_ind - (self._ntime_block - self._ntime_overlap)
            self._next_buf[:,nb_start_ind:nb_start_ind + this_ntime] = intensity

        if start_ind + this_ntime < self._ntime_block:
            self._this_buf[:,start_ind:start_ind + this_ntime] = intensity
            self._ntime_this_buf += this_ntime
        else:
            nt = self._ntime_block - start_ind
            self._this_buf[:,start_ind:] = intensity[:,:nt]
            self._correlated_data_queue.put((self._this_buf_t0, self._this_buf))
            self._this_buf = self._next_buf
            self._ntime_this_buf = nb_start_ind + this_ntime
            self._next_buf = np.zeros((self.nfreq, self._ntime_block),
                                      dtype=np.float32)
            self._this_buf_t0 += ((self._ntime_block - self._ntime_overlap) *
                                  self._delta_t_native)

        self._ntime_absorbed += this_ntime

    def end_stream(self):
        self._correlated_data_queue.put((0, None))


def preprocess_chunk(t0, intensity, weight):
    # Subtract off weighted time mean.
    num = np.sum(intensity * weight, -1)
    den = np.sum(weight, -1)
    den[den == 0] = 1
    intensity -= (num / den)[:,:,None]
    intensity[weight == 0] = 0
    # Combine polarizations.
    intensity = intensity[:,0,:] + intensity[:,1,:]
    # For now ignore weight, even though it is wrong.
    weight = np.zeros_like(weight[:,0,:]) + 255
    preprocess.remove_outliers(intensity, 5)
    intensity -= np.mean(intensity, 1)[:,None]
    preprocess.remove_bad_times(intensity, 2)
    preprocess.remove_noisy_freq(intensity, 3)
    return t0, intensity, weight


class Manager(manager.Manager):

    datasource_class = DataSource

    def __init__(self, vdif_cb_processor, **kwargs):
        # This overwrites some of the defaults in burst_search.
        parameters = dict(PARAMETER_DEFAULTS)
        parameters.update(kwargs)
        super(Manager, self).__init__(
                vdif_cb_processor,
                **parameters
                )

    def preprocess(self, t0, data):
        """No preprocessing.

        Preprocessing is done in the data source where the data arrives in
        small chuncks. This better approximates what will be done in CHIME with
        the incremental DM transform.

        """
        return t0, data


#### ch_frb_io ####
## Adapted from ch_L1mock.datasource

class DiskSource(datasource.DataSource):

    def __init__(self, datadir, **kwargs):
        super(DiskSource, self).__init__(
                source=datadir,
                **kwargs
                )

        self._stream = io.StreamReader(datadir)
        freq = self._stream.freq
        delta_f = np.mean(np.diff(freq))
        if not np.allclose(np.diff(freq), delta_f):
            raise ValueError("Frequencies not uniformly spaced")

        self._delta_f = delta_f / 1e6
        self._freq0 = freq[0] / 1e6
        self._nfreq = len(freq)
        self._delta_t_native = np.median(np.diff(self._stream.time))

        # Don't know these.
        self._mjd = 0
        self._start_time = self._stream.time[0]

        gran = 512   # pretty arbitrary.
        ntime_block = self._block / self._delta_t_native
        ntime_block = int(np.ceil(ntime_block / gran) * gran)
        ntime_overlap = self._overlap / self._delta_t_native
        ntime_overlap = int(np.ceil(ntime_overlap / gran) * gran)
        self._ntime_block = ntime_block
        self._ntime_overlap = ntime_overlap
        self._next_data = np.zeros((self.nfreq, self._ntime_block),
                dtype=np.float32)

        self._nblocks_fetched = 0

    @property
    def nblocks_left(self):
        last_time = self._stream.time[-1]
        nblocks_total = (last_time - self.start_time) // (self._ntime_block -
                self._ntime_overlap)
        return int(nblocks_total) - self.nblocks_fetched

    @property
    def nblocks_fetched(self):
        return self._nblocks_fetched

    def get_next_block_native(self):
        start_of_chunk = self.start_time + (self._ntime_block -
                self._ntime_overlap) * self._delta_t_native * self.nblocks_fetched
        start_of_next_chunk = self.start_time + (self._ntime_block -
                self._ntime_overlap) * self._delta_t_native * (self.nblocks_fetched + 1)
        end_of_chunk = start_of_chunk + self._ntime_block * self._delta_t_native
        #start_of_chunk = self.last_time0 + self.delta_t_native * self.ntime_chunk
        #end_of_chunk = self.last_time0 + self.delta_t_native * self.ntime_chunk * 2
        # Read a bit extra, since we will be buffering the overlap anyway.
        last_ind = np.searchsorted(self._stream.time, end_of_chunk +
                self._ntime_overlap * self._delta_t_native / 10)
        ntime_read = last_ind - self._stream.current_time_ind

        datasets = self._stream.yield_chunk(ntime_read)
        time = datasets['time']
        intensity = datasets['intensity']
        weights = datasets['weight']
        data, weights = _format_intensity_weights(intensity, weights)

        # Copy data into buffered data and new buffer.
        this_data = self._next_data
        self._next_data = np.zeros((self.nfreq, self._ntime_block),
                dtype=np.float32)


        _align_copy_data(time, data, start_of_chunk, self._delta_t_native,
                this_data)
        _align_copy_data(time, data, start_of_next_chunk, self._delta_t_native,
                self._next_data)

        self._nblocks_fetched += 1
        return start_of_chunk, this_data


def _align_copy_data(time, data, buf_start, buf_dt, buf):
    if len(data.shape) != 2 or len(buf.shape) != 2:
        raise ValueError()
    if data.shape[0] != buf.shape[0]:
        raise ValueError()
    if data.shape[1] != len(time):
        raise ValueError()

    buf_ntime = buf.shape[1]

    inds = np.round((time - buf_start) / buf_dt).astype(int)
    data_start = np.where(inds >= 0)[0][0]
    data_end = np.where(inds < buf_ntime)[0][-1]
    # XXX horribly inefficient.
    buf[:,inds[data_start:data_end]] = data[:,data_start:data_end]



def _format_intensity_weights(intensity, weights):
    # XXX -LIAM- This assumes uint8 weights scaled to 128. I don't think this is true
    # anymore.
    in_shape = intensity.shape
    data = np.empty((in_shape[0], in_shape[2]), dtype=np.float32)
    weights_new = np.empty_like(data)
    data[:,:] = intensity[:,0,:]
    data += intensity[:,1,:]
    # Taking care not to overflow.
    weights_new[:] = weights[:,0,:]
    weights_new += weights[:,1,:]
    # Mask data with less than 50% weight.
    # XXX
    #weights_new[np.logical_or(weights[:,0,:] < 128, weights[:,1,:] < 128)] = 0
    # Normalize.
    weights_new /= 255 * 2
    return data, weights_new


class DiskManager(manager.Manager):

    datasource_class = DiskSource

    def __init__(self, datadir, **kwargs):
        # This overwrites some of the defaults in burst_search.
        parameters = dict(PARAMETER_DEFAULTS)
        parameters.update(kwargs)
        super(DiskManager, self).__init__(
                datadir,
                **parameters
                )


