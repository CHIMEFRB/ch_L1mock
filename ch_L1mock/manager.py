"""
Driver class for the L1 mock up.
"""

from burst_search import preprocess

from ch_L1mock import datasource, L0, utils
import ch_L1mock.tests.data as testdata


class Manager(object):

    def __init__(self, conf):
        self._conf = conf
        # Parse global parameters.
        self._ntime_chunk = conf['ntime_chunk']

        # Initialize some internal variables.
        self._daemon_threads = []

        # Set up the bits and pieces.
        self._configure_source()
        self._configure_dedisperser()
        self._configure_preprocessing()
        self._configure_postprocessing()
        self._configure_actions()

    def _configure_source(self):
        parameters = self._conf['source']
        ds_type = parameters['type']
        if ds_type == 'disk':
            data_dir = parameters['data_dir']
            if data_dir == "TESTDATA":    # Special value.
                data_dir = testdata.TEST_DATA_DIR
            ds = datasource.DiskSource(
                    data_dir,
                    ntime_chunk=self._ntime_chunk,
                    )
        elif ds_type == 'vdif':
            import ch_vdif_assembler
            vdif_source_pars = pars['vdif_source']
            vdif_source_type = vdif_source_pars.pop('type')
            if vdif_source_type == 'network':
                stream = ch_vdif_assembler.make_network_stream(
                        **vdif_source_pars)
            elif vdif_source_type == 'simulate':
                stream = ch_vdif_assembler.make_simulated_stream(
                        **vdif_source_pars)
            elif vdif_source_type == 'moose-acq':
                stream = ch_vdif_assembler.moose_acquisition(
                        **vdif_source_pars)
            elif vdif_source_type == 'file-list':
                stream = ch_vdif_assembler.make_file_stream(
                        **vdif_source_pars)
            else:
                msg = "Invalid vdif source type: %s."
                raise ValueError(msg % vdif_source_type)
            assembler = ch_vdif_assembler.assembler()
            p = L0.CallBackCorrelator(
                    nframe_integrate=parameters['nframe_integrate'],
                    )
            ds = datasource.vdifSource(p, ntime_chunk=ntime_chunk)
            assembler.register_processor(p)
            thread = utils.ExceptThread(
                target=assembler.run,
                args=(stream,),
                )
            self._daemon_threads.append(thread)
        else:
            msg = "Data source type %s nor supported."
            raise ValueError(msg % ds_type)
        self._datasource = ds

    def _configure_dedisperser(self):
        pass

    def _configure_preprocessing(self):
        #self._dedisperser.preprocess_data = preprocess_chunk
        pass

    def _configure_postprocessing(self):
        #self._dedisperser.process_triggers = something something
        pass

    def _configure_actions(self):
        pass

    def _start(self):
        for t in self._daemon_threads:
            t.start()

    def _finalize(self):
        for t in self._daemon_threads:
            t.check_join()
        self._datasource.finalize()

    def run(self):
        self._start()

        while True:
            # Check the status of the daemon threads.
            for t in self._daemon_threads:
                t.check()
            # Get some data.
            try:
                data, mask = self._datasource.yield_chunk(timeout=0.1)
            except StopIteration:
                # We're done.
                break
            except datasource.NoData:
                # No data ready. Keep waiting.
                continue
            # We have data! Call in the dedisperser.
            #self._dedisperser.run(data)

        self._finalize()


def preprocess_chunk(data, mask):
    subtract_masked_mean(data, mask)
    # The following mostly ignore the mask and don't update it.
    preprocess.remove_outliers(data, 5)
    subtract_masked_mean(data, mask)
    preprocess.remove_bad_times(data, 2)
    preprocess.remove_noisy_freq(data, 3)


def subtract_masked_mean(data, mask):
    num = np.sum(data * mask, -1)
    den = np.sum(mask, -1)
    den[den == 0] = 1
    data -= (num / den)[:,None]
    data[mask == 0] = 0




