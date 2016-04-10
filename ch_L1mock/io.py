"""
IO for intensity data.

"""

import os
from os import path
import warnings
import logging

import numpy as np
import h5py
import bitshuffle.h5 as bshufh5


logger = logging.getLogger(__name__)

# Default chunk and file size.
CHUNKS = (64, 2, 256)
NTIME_PER_FILE = CHUNKS[2] * 64


# Dataset definitions.
DATASETS = {
    # 'time' is a special axis defining dataset. Must have units seconds since
    # it is used for filenames.
    'index_map/time' : {
        'dtype' : np.float64,
        'chunks' : (CHUNKS[2],),
        },
    'intensity' : {
        'dtype' : np.float32,
        'axis' : ['freq', 'pol', 'time'],
        'chunks' : CHUNKS,
        'compression' : bshufh5.H5FILTER,
        'compression_opts' : (0, bshufh5.H5_COMPRESS_LZ4),
        },
    'weight' : {
        'dtype' : np.uint8,
        'axis' : ['freq', 'pol', 'time'],
        'chunks' : CHUNKS,
        'compression' : bshufh5.H5FILTER,
        'compression_opts' : (0, bshufh5.H5_COMPRESS_LZ4),
        },
    }


class StreamWriter(object):

    def __init__(self, outdir, freq, pol, attrs=None):
        self._outdir = outdir
        self._freq = freq
        self._nfreq = len(freq)
        self._pol = pol
        self._npol = len(pol)
        if attrs is None:
            attrs = {}
        self._attrs = attrs

        # For now these are statically defined.
        self._ntime_per_file = NTIME_PER_FILE
        self._ntime_block = CHUNKS[2]
        self._datasets = dict(DATASETS)
        assert self._ntime_per_file % self.ntime_block == 0

        # Initialize dataset buffers.
        self._buffers = {}
        datasets = dict(self._datasets)
        time_info = datasets.pop('index_map/time')
        self._buffers['index_map/time'] = np.empty(self.ntime_block,
                                                   dtype=time_info['dtype'])
        for name, info in datasets.items():
            if info['axis'] != ['freq', 'pol', 'time']:
                msg = "Only ('freq', 'pol', 'time') datasets supported."
                raise NotImplementedError(msg)
            self._buffers[name] = np.empty(
                    (self._nfreq, self._npol, self.ntime_block),
                    dtype = info['dtype']
                    )
            if self.ntime_block % info['chunks'][2]:
                msg = "Integer number of chunks must fit into buffer."
                raise ValueError(msg)
            # TODO Check sanity of other chunk dimensions.

        # Buffers initially empty.
        self._ntime_buffer = 0
        # Initialize output.
        self._file = None
        self._t0 = None    # Offset for file names.
        if not path.isdir(outdir):
            os.mkdir(outdir)


        # Ensure that warnings only issued once.
        self._alignment_warned = False

    def __del__(self):
        self.finalize()

    @property
    def ntime_block(self):
        """Target write size. The size of the buffer when full."""
        return self._ntime_block

    @property
    def ntime_buffer(self):
        """Current number of times currently in the buffer."""
        return self._ntime_buffer

    @property
    def ntime_current_file(self):
        """Number of times in current file."""
        if self._file is None:
            return 0
        else:
            return len(self._file['index_map/time'])

    @property
    def ntime_per_file(self):
        return self._ntime_per_file

    def absorb_chunk(self, **kwargs):
        """Currently the number of time samples much add up to ntime.
        """

        time = kwargs.pop('time')
        ntime = len(time)

        for name, data in kwargs.items():
            if data.shape != (self._nfreq, self._npol, ntime):
                msg = "Inconsistent dimensions for dataset %s" % name
                raise ValueError(msg)
        kwargs['index_map/time'] = time

        ntime_consumed = 0
        while ntime_consumed < ntime:
            ntime_remaining = ntime - ntime_consumed
            if self.ntime_buffer == 0 and ntime_remaining >= self.ntime_block:
                # If the buffers are empty and ntime is bigger than the buffer
                # size, do a direct write.
                to_write = (ntime_remaining
                            - (ntime_remaining % self.ntime_block))
                to_write = min(to_write,
                               self._ntime_per_file - self.ntime_current_file)
                self._append_data_disk(
                        ntime_consumed,
                        ntime_consumed + to_write,
                        **kwargs
                        )
                ntime_consumed = ntime_consumed + to_write
            else:
                # Add data to buffers.
                to_buffer = min(self.ntime_block - self.ntime_buffer,
                                ntime_remaining)
                self._append_data_buffers(
                        ntime_consumed,
                        ntime_consumed + to_buffer,
                        **kwargs
                        )
                ntime_consumed = ntime_consumed + to_buffer

    def flush(self):
        if (self.ntime_buffer != self.ntime_block
            and not self._alignment_warned):
            msg = ("Flushing buffers that are not full. Expect alignment"
                   " issues and performance degradation.")
            logger.warning(msg)
            self._alignment_warned = True
        self._append_data_disk(0, self.ntime_buffer, **self._buffers)
        self._ntime_buffer = 0

    def finalize(self):
        # Do nothing if this has already been called.
        if hasattr(self, '_datasets'):
            # Suppress warning if the buffers aren't full.
            self._alignment_warned = True
            self.flush()
            if self._file:
                self._file.close()
            # The following does two things: releases memory which is nice, but
            # more importantly invalidates the instance.
            del self._buffers
            del self._datasets

    def _initialize_file(self, first_time):
        # Files are named with their starting time relative to beginning of
        # acquisition.
        if self._t0 is None:
            self._t0 = first_time
        first_time -= self._t0
        fname = '%08d.h5' % int(round(first_time))
        fname = path.join(self._outdir, fname)
        # Open file and write non-time-dependant datasets.
        f = h5py.File(fname, mode='w')
        for name, value in self._attrs.items():
            f.attrs[name] = value
        # Index map
        im = f.create_group('index_map')
        im.create_dataset('pol', data=self._pol)
        im.create_dataset('freq', data=self._freq)
        # Initialize time dependant datasets.
        datasets = dict(self._datasets)
        time_dset_info = datasets.pop('index_map/time')
        f.create_dataset(
                'index_map/time',
                shape=(0,),
                maxshape=(None,),
                dtype=time_dset_info['dtype'],
                chunks=time_dset_info['chunks'],
                )
        for dset_name, dset_info in datasets.items():
            dset = f.create_dataset(
                    dset_name,
                    shape=(self._nfreq, self._npol, 0),
                    maxshape=(self._nfreq, self._npol, None),
                    dtype=dset_info['dtype'],
                    chunks=dset_info['chunks'],
                    compression=dset_info['compression'],
                    compression_opts=dset_info['compression_opts'],
                    )
            dset.attrs['axis'] = dset_info['axis']
        self._file = f

    def _append_data_disk(self, start, stop, **kwargs):
        if self._file is None:
            first_time = kwargs['index_map/time'][start]
            self._initialize_file(first_time)
        ntime_disk = self.ntime_current_file

        ntime = stop - start
        time = kwargs.pop('index_map/time')
        self._file['index_map/time'].resize((ntime_disk + ntime,))
        self._file['index_map/time'][ntime_disk:] = time[start:stop]

        for name, data in kwargs.items():
            dset = self._file[name]
            dset.resize((self._nfreq, self._npol, ntime_disk + ntime))
            dset[...,ntime_disk:] = data[...,start:stop]
        if ntime_disk + ntime >= self._ntime_per_file:
            self._file.close()
            self._file = None

    def _append_data_buffers(self, start, stop, **kwargs):
        ntime = stop - start
        for name, data in kwargs.items():
            buf = self._buffers[name]
            buf_sl = np.s_[...,self.ntime_buffer:self.ntime_buffer + ntime]
            buf[buf_sl] = data[...,start:stop]
        self._ntime_buffer += ntime
        if self.ntime_buffer == self.ntime_block:
            self.flush()
