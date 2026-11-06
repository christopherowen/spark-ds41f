"""CPU-only ordering/statistics shared by the paired hardware screen."""
import math
import statistics


def arm_order(arms, block):
    """A Latin rotation, reversing whole cycles rather than cancelling pairs."""
    offset = block % len(arms)
    order = list(arms[offset:]) + list(arms[:offset])
    if (block // len(arms)) % 2:
        order.reverse()
    return order


def interval(values):
    if len(values) != 8 or not all(math.isfinite(v) for v in values):
        raise ValueError('the predeclared interval requires eight finite blocks')
    mean = statistics.fmean(values)
    half = 2.365 * statistics.stdev(values) / math.sqrt(8)  # two-sided t(7), 95%
    return [mean - half, mean + half]
