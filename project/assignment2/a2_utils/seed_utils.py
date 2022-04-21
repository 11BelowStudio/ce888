# just contains stuff for an rng seed.

from __future__ import annotations

from numpy.random import Generator, RandomState, default_rng
from typing import Final, Union, Optional


__all__ = ["seed", "rng", "rng_state", "RNG"]
"_seed is not allowed to be imported."

_seed: Final[int] = 42
"Using the meaning of life, the universe, and everything as the seed for RNG"


def seed() -> int:
    """
    returns _seed (in a readonly manner)
    :return: the seed.
    """
    return _seed


def rng() -> Generator:
    """
    Creates a new numpy random generator with a seed of 42.
    :return: a new numpy random generator with a seed of 42
    """
    return default_rng(seed=_seed)


def rng_state() -> RandomState:
    """
    Creates a new numpy randomstate with a seed of 42
    :return: a new numpy randomstate with a seed of 42
    """
    return RandomState(seed=_seed)


RNG = Optional[
                Union[
                    int,
                    RandomState
                ]
            ]
"type alias for the randomstate parameter taken by a bunch of sklearn things"
