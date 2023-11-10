import ipdb
import traceback

def with_debugger(orig_fn):
    def new_fn(*args, **kwargs):
        try:
            return orig_fn(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            ipdb.post_mortem()

    return new_fn


class SparseExceptionLogger():
    def __init__(self, logger, extra_starts=None):
        self.known_errors = set([])
        self.logger = logger
        self.extra_starts = extra_starts if extra_starts is not None else []

    def __call__(self, prefix_msg, exception):
        self.log(prefix_msg, exception)

    def log(self, prefix_msg, exception):
        msg = str(exception)
        msg = self.preprocess_msg(msg)

        if msg not in self.known_errors:
            self.known_errors.add(msg)
            self.logger.exception(prefix_msg + " (LOGGING ONCE)")

    def preprocess_msg(self, msg):
        starts = ["CUDA out of memory.", "one of the variables needed for gradient computation has been modified by an inplace operation"]
        starts += self.extra_starts
        for start in starts:
            if msg.startswith(start):
                return start
        return msg
