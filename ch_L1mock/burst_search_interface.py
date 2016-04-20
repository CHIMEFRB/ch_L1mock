"""
Back end FRB search engine using the Burst Search package.

"""

from Queue import Queue

import numpy as np
from burst_search import datasource, manager

import constants


class DataSource(datasource.DataSource):

    def __init__(self, vdif_cb_processor, **kwargs):
        super(DataSource, self).__init__(
                source=vdif_cb_processor,
                **kwargs
                )

        vdif_cb_processor.add_callback(self.absorb_chunk)

        self._correlated_data_queue = Queue()
        self._nblocks_fetched = 0

    @property
    def nblocks_left(self):
        return self._correlated_data_queue.qsize()

    @property
    def nblocks_fetched(self):
        return self._nblocks_fetched

    def get_next_block_native(self):
        t0, data = self._correlated_data_queue.get()
        self._nblocks_fetched += 1
        return t0, data

    def absorb_chunk(self, time, intensity, weight):
        pass


class Manager(manager.Manager):

    datasource = DataSource

    def preprocess(self, t0, data):
        """No preprocessing.

        Preprocessing is done in the data source where the data arrives in
        small chuncks. This better approximates what will be done in CHIME with
        the incremental DM transform.

        """
        return t0, data
