import logging
from timeit import default_timer as timer
from datetime import timedelta


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timer()
        result = func(*args, **kwargs)
        elapsed_time = timer() - start_time
        logging.debug(f"[Profiler] {func.__name__}: {timedelta(seconds=elapsed_time)}")
        return result

    return wrapper
