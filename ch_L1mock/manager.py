"""
Driver classes for the L1 mock up.
"""

import bonsai

from ch_L1mock import datasource, preprocess, postprocess, action, L0, utils
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
        self._configure_action()

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
        parameters = dict(self._conf['dedisperse'])
        datasource = self._datasource
        parameters['nchan'] = datasource.nfreq
        parameters['nt_data'] = datasource.ntime_chunk
        parameters['dt_sample'] = datasource.delta_t
        # XXX Check that these are consistent with bonsai defn.
        print datasource.delta_f, datasource.nfreq
        parameters['freq_lo_MHz'] = (datasource.freq0
                + datasource.nfreq * datasource.delta_f) / 1e6
        parameters['freq_hi_MHz'] = datasource.freq0 / 1e6
        print parameters
        self._dedisperser = ConfigurableDedisperser(parameters)

    def _configure_preprocessing(self):
        tasks = self._conf['preprocess']
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task_spec in tasks:
            task_name = task_spec.pop('type')
            task = preprocess.INDEX[task_name](self._dedisperser, **task_spec)
            self._dedisperser.preprocess_tasks.append(task)

    def _configure_postprocessing(self):
        tasks = self._conf['postprocess']
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task_spec in tasks:
            task_name = task_spec.pop('type')
            task = postprocess.INDEX[task_name](self._dedisperser, **task_spec)
            self._dedisperser.postprocess_tasks.append(task)

    def _configure_action(self):
        tasks = self._conf['action']
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task_spec in tasks:
            task_name = task_spec.pop('type')
            task = action.INDEX[task_name](self._dedisperser, **task_spec)
            self._dedisperser.action.append(task)

    def _start(self):
        for t in self._daemon_threads:
            t.start()
        self._dedisperser.spawn_slave_threads()

    def _finalize(self):
        for t in self._daemon_threads:
            t.check_join()
        self._datasource.finalize()
        self._dedisperser.terminate()

    def run(self):
        self._start()

        while True:
            # Check the status of the daemon threads.
            for t in self._daemon_threads:
                t.check()
            # Get some data.
            try:
                data, weights = self._datasource.yield_chunk(timeout=0.1)
            except StopIteration:
                # We're done.
                break
            except datasource.NoData:
                # No data ready. Keep waiting.
                continue
            # We have data! Call in the dedisperser.
            self._dedisperser.run(data, weights)

        self._finalize()


class ConfigurableDedisperser(bonsai.Dedisperser):

    @property
    def preprocess_tasks(self):
        return self._preprocess_tasks

    @property
    def postprocess_tasks(self):
        return self._postprocess_tasks

    @property
    def action(self):
        return self._action

    def __init__(self, config, ibeam=0, read_analytic_variance=True):
        self._preprocess_tasks = []
        self._postprocess_tasks = []
        self._action = []

    def preprocess_data(self, data, weights):
        for t in self.preprocess_tasks:
            t(data, weights)

    def process_triggers(self, itree, triggers):
        events = []
        for t in self.postprocess_tasks:
            events += t(itree, triggers)
        for t in self.action:
            t(itree, events)
