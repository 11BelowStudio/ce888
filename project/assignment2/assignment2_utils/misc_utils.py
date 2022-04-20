"""
miscellaneous utilities

"""

from __future__ import annotations

from itertools import chain
import numpy as ndarray

from typing import List, Tuple, TextIO, Dict, Iterable, Iterator, Union, Optional, Literal, NoReturn, TypeVar, Any

T = TypeVar("T")
"a generic type"



def x_else(
        x: Optional[T],
        fallback: T
) -> T:
    """
    if x is none, return fallback. otherwise return x.
    :param x:
    :param fallback:
    :return:
    """
    if x is None:
        return fallback
    return x


def chain_1(
        the_iter: Iterable[T],
        add_this: T
) -> Iterator[T]:
    """
    shortcut for appending the singular 'add this' to the iterable 'the_iter'
    :param the_iter: iterator of type T
    :param add_this: the non-iterable thing we want to add to the iterable's iterator
    :return: itertools.chain(the_iter, [add_this])
    """
    return chain(the_iter, [add_this])


def iter_is_none(
        the_iter: Optional[Iterable[Union[T, None]]]
) -> bool:
    """
    shortcut for seeing if an iterator is actually None or is empty
    :param the_iter: an iterable that may or may not contain stuff/exist
    :return: true if 'the_iter' is None, empty, or only contains Nones
    """
    return (
                   the_iter is None
           ) or (
               not any(
                   i is not None
                   for i in the_iter
               )
           )
