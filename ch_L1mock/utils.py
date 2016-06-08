"""
Miscellaneous utilities.
"""

import sys
import threading
import Queue



class ExceptThread(threading.Thread):
    """Thread that can communicate exceptions."""

    def __init__(self, *args, **kwargs):
        super(ExceptThread, self).__init__(*args, **kwargs)
        self._error_bucket = Queue.Queue()
        self.daemon = True

    def run(self):
        try:
            super(ExceptThread, self).run()
        except:
            self._error_bucket.put(sys.exc_info())

    def check(self):
        try:
            err_info = self._error_bucket.get(False)
            raise err_info[1], None, err_info[2]
        except Queue.Empty:
            pass

    def check_join(self):
        while True:
            self.join(0.1)
            if not self.is_alive():
                self.check()
                break


