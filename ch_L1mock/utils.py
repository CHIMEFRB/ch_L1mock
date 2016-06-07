"""
Miscellaneous utilities.
"""

import sys
import threading
import Queue


class ExcThread(threading.Thread):
    """Thread that communicates exceptions."""

    def __init__(self, bucket=None, **kwargs):
        if not hasattr(bucket, 'put'):
            raise ValueError("*bucket* parameter should be a Queue.")
        super(ExcThread, self).__init__(**kwargs)
        self._bucket = bucket

    def run(self):
        try:
            super(ExcThread, self).run()
        except:
            self._bucket.put(sys.exc_info())


def start_daemon_thread(target, args=(), kwargs=None):

    if kwargs is None:
        kwargs = {}

    bucket = Queue.Queue()

    thread = ExcThread(bucket=bucket, target=target, args=args, kwargs=kwargs)
    thread.daemon = True

    thread.start()
    return thread, bucket





