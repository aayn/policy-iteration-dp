from itertools import islice, product
from functools import reduce, lru_cache
from collections import deque
from math import factorial, exp


def mapt(f, *iterables):
    return tuple(map(f, *iterables))


def first(iterable, default=None):
    return next(iter(iterable), default)


def second(iterable, default=None):
    k = iter(iterable)
    next(k, default)
    return next(k, default)


def product_t(*iters):
    "Tuple of cartesian product."
    return tuple(product(*iters))


@lru_cache(maxsize=None)
def poisson(n, l):
    "Probability mass function for Poisson distribution."
    return (pow(l, n) / factorial(n)) * exp(-l)