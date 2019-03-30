from itertools import islice, product
from functools import reduce, lru_cache
from collections import deque
import operator as op
from math import factorial, exp


def I(x):
    return x


def mapt(f, *iterables):
    return tuple(map(f, *iterables))


def mapl(f, *iterables):
    return list(map(f, *iterables))


def first(iterable, default=None):
    return next(iter(iterable), default)


def tail(n, iterable):
    "Return an iterator over the last n items."
    # tail(3, 'ABCDEFG') --> E F G
    return iter(deque(iterable, maxlen=n))


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.
    If no true value is found, returns *default*
    If *pred* is not None, returns the first item
    for which pred(item) is true."""
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def first_n(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def cart_prod(*iters):
    "Tuple of cartesian product."
    return tuple(product(*iters))


def make_table(*iters, operation=op.mul, initial=1):
    """Given n iterables, make an n-dimensional table.

    operation: Combines into a single table value using the operation.
    """
    return mapt(lambda x: reduce(lambda acc, y: operation(acc, y), initial),
                cart_prod(*iters))


@lru_cache(maxsize=None)
def poisson(n, l):
    return (pow(l, n) / factorial(n)) * exp(-l)