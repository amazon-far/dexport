"""Low-jitter waiting for the deploy loop (UMI's ``precise_wait``)."""

import time


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic) -> None:
    """Wait until ``time_func()`` reaches ``t_end``: sleep until ``slack_time`` before it, then
    spin, since ``time.sleep`` alone can overshoot by a scheduler tick."""
    t_wait = t_end - time_func()
    if t_wait > 0:
        if t_wait > slack_time:
            time.sleep(t_wait - slack_time)
        while time_func() < t_end:
            pass
