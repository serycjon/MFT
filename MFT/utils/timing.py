from timeit import default_timer as timer
import torch
import logging
import inspect
import numpy as np


class time_measurer():
    def __init__(self, units='ms', desc=None):
        self.start_time = timer()
        self.units = units
        self.desc = desc

    def __call__(self):
        return self.elapsed()

    def elapsed(self):
        value = float(timer() - self.start_time)
        if self.units == 'ms':
            value = float(f'{(1000 * value):.1f}')
        return value

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        caller_name = inspect.getmodule(inspect.stack()[1][0]).__name__
        logger = logging.getLogger(caller_name)
        ms_elapsed = self.elapsed()
        logger.debug(f"{self.desc}: {ms_elapsed}ms")


class cuda_time_measurer():
    """ https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    https://auro-227.medium.com/timing-your-pytorch-code-fragments-e1a556e81f2 """

    def __init__(self, units=None):
        # self.start_time = timer()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.units = units

        self.start_event.record()

    def __call__(self):
        self.end_event.record()
        torch.cuda.synchronize()  # this is intentionally and correctly here, AFTER the end_event.record()
        value = self.start_event.elapsed_time(self.end_event)
        assert self.units == 'ms'
        return value


class general_time_measurer():
    def __init__(self, name=None, active=True, cuda_sync=True, start_now=True):
        self.name = name
        self.active = active
        self.cuda_sync = cuda_sync
        self.durations_ms = []
        self.is_started = False

        if start_now:
            self.start()

    def start(self):
        if not self.active:
            return

        assert not self.is_started, "You must first stop the timer to record again"

        if self.cuda_sync:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = timer()

        self.is_started = True

    def stop(self):
        if not self.active:
            return

        if self.cuda_sync:
            self.end_event.record()
            torch.cuda.synchronize()  # this is intentionally and correctly here, AFTER the end_event.record()
            duration_ms = self.start_event.elapsed_time(self.end_event)
        else:
            duration_ms = 1000 * float(timer() - self.start_time)

        self.durations_ms.append(duration_ms)

    def report(self, reduction='mean'):
        if not self.active:
            return

        if self.is_started:
            self.stop()

        if reduction == 'mean':
            value = np.nanmean(self.durations_ms)
            name = f'mean_{self.name}'
        elif reduction == 'sum':
            value = np.nansum(self.durations_ms)
            name = f'total_{self.name}'

        if len(self.durations_ms) == 1:
            name = self.name

        caller_name = inspect.getmodule(inspect.stack()[1][0]).__name__
        logger = logging.getLogger(caller_name)
        logger.debug(f"{name}: {value:.2f}ms")
