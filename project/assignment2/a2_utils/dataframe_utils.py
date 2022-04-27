"""
Imports dataframes and such

"""

from __future__ import annotations

import pandas as pd
import numpy as np

import os

from typing import List, Tuple, TextIO, Dict, Iterable, Iterator, Union, Optional, Literal, NoReturn, TypeVar, Any
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from pandas.core.indexes.base import Index

import doctest
import sys

import dataclasses
from functools import cached_property

from itertools import chain

from enum import Enum, auto

import pickle


from assignment2.a2_utils.seed_utils import *

from assignment2.a2_utils.misc_utils import *

T_INDEX = Union[np.ndarray, Index]
"type alias for potential datatypes that could be used to obtain data from the dataframe by index"

"""
STUFF THAT'S USED WHEN CHECKING IMPORTS
"""


def check_version(
        actual_version_string: str,
        min_version: Tuple[int, int, int],
        vocal: bool = False,
        write_to: TextIO = sys.stderr
) -> bool:
    """
    Attempts to ensure that a version string (from, say, 'pd.__version__') meets a minimum version
    (pass (1,4,1) to match version 1.4.1 or greater)

    :param actual_version_string: a .__version__ string
    :param min_version: a minimum version as a tuple
    :param vocal: should there be an error message if the wrong version is given?
    :param write_to: Where should the error message (if vocal and there is an error) be written to?
    :return: true if the given actual version is no smaller than the min version.

    >>> check_version("1.4.1",(1,4,1))
    True
    >>> check_version("1.4.2",(1,4,1))
    True
    >>> check_version("1.4.0",(1,4,1))
    False
    >>> check_version("1.3.9",(1,4,1), vocal=True, write_to=sys.stdout)
    Version 1.3.9 is below required version 1.4.1!
    False
    """

    actual_version_tup: Tuple[int, ...] = tuple(
        int(i) for i in actual_version_string.split(".")
    )
    for actual_ver, min_ver in zip(actual_version_tup, min_version):
        if actual_ver > min_ver:
            return True
        elif actual_ver < min_ver:
            if vocal:
                print(
                    f"Version {actual_version_string} "
                    f"is below required version {'.'.join(str(i) for i in min_version)}!",
                    file=write_to
                )
            return False
        else:
            continue
    return True


assert check_version(pd.__version__, (1, 4, 1), vocal=True)

"""
###

   STUFF THAT'S USED WHEN READING A DATAFRAME

###
"""


def npz_to_dict(
        npz_filename: str,
        **kwargs
) -> Dict[str, np.ndarray]:
    """
    Puts the given NPZ file into a dictionary of {table name, ndarray}.
    :param npz_filename: the filename of the NPZ file that we want to put into a dictionary.
    :param kwargs: kwargs from https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load.
    DO NOT INCLUDE A 'mmap_mode' KWARG!!!
    :return: The data from the given npz file in a dictionary.
    """
    data_dict: Dict[str, np.ndarray] = {}
    if kwargs is None:
        kwargs = {}
    with np.load(npz_filename, mmap_mode="r", **kwargs) as npz:
        for f in npz.files:
            data_dict[f] = npz[f]
    return data_dict


def x_to_dataframe(
        x_data: np.ndarray,
        row_major=True,
        x_prefix: str = "x"
) -> pd.DataFrame:
    """
    Converts the 'x' ndarray into a pandas dataframe.
    :param x_data: the ndarray containing all of the data
    :param row_major: is this ndarray held in row-major order? [[item a data], [item b data], ... ]
    :param x_prefix: prefix to put on the names of all of the x columns
    :return: a dataframe holding the given x data.
    """
    if row_major:
        x_data: np.ndarray = x_data.T
    x_df: pd.DataFrame = pd.DataFrame.from_dict({f"{x_prefix}{i}": x_data[i] for i in range(x_data.shape[0])})

    return turn_01_columns_into_int(x_df)


def add_everything_but_x_to_copy_of_dataframe(
        original_df: pd.DataFrame,
        the_data_dict: Dict[str, np.ndarray],
        dont_add: Union[str, Iterable[str]] = frozenset('x')
) -> pd.DataFrame:
    """
    Adds everything in the npz file apart from the given 'dont_add'
    tables to the dataframe.
    Assumes that these other tables have the same shape of (whatever, 1).
    :param original_df: the original dataframe that shall be copied and have stuff added to it.
    :param the_data_dict: The data file with the data to be added to the DataFrame
    :param dont_add: the identifier(s) of the columns that must not be added to the DataFrame.
    :return: a copy of the original dataframe, with the data from every table BESIDES the 'dont add' tables
    from the given file added to it.
    """

    the_df = original_df.copy()
    if dont_add in the_data_dict.keys():
        dont_add = [dont_add]
    for k, v in the_data_dict.items():
        if k in dont_add:
            continue
        the_df[k] = pd.DataFrame(v)

    return turn_01_columns_into_int(the_df)


def turn_01_columns_into_int(
        dataframe_to_edit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finds all of the columns that just contain values of 0 and 1,
    and converts all of those columns to ints.

    Dataframe will have an '01' and 'not_01' attr added to it.
    Labels for series that only contain values 0 and 1 will be in the '01' tuple
    Labels for every other series will be in the 'not_01' tuple

    MODIFIES THE GIVEN DATAFRAME!
    :param dataframe_to_edit: the dataframe that is being edited
    :return: The modified dataframe.
    DOES NOT COPY THE GIVEN ORIGINAL DATAFRAME.

    >>> import pandas as pd
    >>> check_version(pd.__version__, (1,4,1))
    True
    >>> before: pd.DataFrame = pd.DataFrame.from_dict(data={"int01":[0,1,1,0],"flt01":[0.0, 1.0, 0.0, 1.0], "intNo": [-1,0,1,2], "fltNo":[-1.0, 0.0, 1.0, 2.0], "intNan": [0,1,None,0], "fltNan":[0.0,1.0,None,0.0]})
    >>> before_types = before.dtypes.values
    >>> after: pd.DataFrame = turn_01_columns_into_int(before.copy())
    >>> after_types = after.dtypes.values
    >>> print(after_types[0])
    uint8
    >>> print(after_types[1])
    uint8
    >>> print(f"{before_types[2] == after_types[2]} {before_types[3] == after_types[3]} {before_types[4] == after_types[4]} {before_types[5] == after_types[5]}")
    True True True True
    >>> after.attrs['01']
    ('int01', 'flt01')
    >>> after.attrs['not_01']
    ('intNo', 'fltNo', 'intNan', 'fltNan')
    """
    cols_01: List[str] = []
    not_01: List[str] = []
    for c in dataframe_to_edit.columns:
        # if dataframe_to_edit[c].dtype == np.uint8:
        #    continue
        if dataframe_to_edit[c].isin([0, 1]).all():
            dataframe_to_edit[c] = dataframe_to_edit[c].astype(np.uint8)
            cols_01.append(c)
        else:
            not_01.append(c)
    dataframe_to_edit.attrs["01"] = tuple(cols_01)
    dataframe_to_edit.attrs["not_01"] = tuple(not_01)
    return dataframe_to_edit


def process_counterfactuals(
        df: pd.DataFrame,
        t: str = "t",
        y_factual: str = "yf",
        new_counterfactual_t: str = "tcf",
        y_counterfactual: Optional[str] = None,
        ite: Optional[str] = None,
        t0_name: Optional[str] = "t0",
        t1_name: Optional[str] = "t1"
) -> pd.DataFrame:
    """
    Processes the counterfactual info in the dataframe, adding it to the dataframe.
    EDITS THE 'df' DATAFRAME!!!!
    :param df: the dataframe.
    :param t: name of the column containing 'treatment'
    :param y_factual: name of the column with factual T data
    :param new_counterfactual_t: name of the NEW column to put 'counterfactual T' into
    :param y_counterfactual: name of the column with counterfactual Y. Replace with 'None' if it doesn't exist.
    :param ite: name of the column with individual treatment effects. Replace with 'None' if it doesn't exist.
    :param t0_name: name of the NEW column to move known 't0' into (only used if ycf and ITE are given)
    :param t1_name: name of the NEW column to move known 't1' into (only used if ycf and ITE are given)
    :return: dataframe with the counterfactuals processed and such
    """

    # adding a new 'tcf' column, containing the inverse of 't'
    df[new_counterfactual_t] = np.choose(df[t].values, [1, 0])

    # and now, processing the counterfactuals (if they exist)
    if y_counterfactual is not None:
        # copying t0 results into t0 column
        df[t0_name] = np.choose(df[t].values, [df[y_factual].values, df[y_counterfactual].values])

        # copying t1 results into t1 column
        df[t1_name] = np.choose(df[t].values, [df[y_counterfactual].values, df[y_factual].values])

    # and then returning the dataframe (ensuring 01 values are preserved as 01)
    return turn_01_columns_into_int(df)


"""
###

   stuff that's relevant after we have opened the dataframe
   
###
"""


def isolate_these_columns(
        full_dataframe: pd.DataFrame,
        special_columns: Iterable[str] = ("e","tcf", "ycf", "ite", "t0", "t1"),
        remove_those_columns: bool = True,
        copy: bool = True
) -> pd.DataFrame:
    """
    shortcut for making a copy of a dataframe excluding some columns. Easier than typing .loc every time
    :param full_dataframe: the full dataframe
    :param special_columns: columns to isolate
    :param remove_those_columns:
        if true, we return a dataframe containing everything BUT those columns.
        Otherwise, return a dataframe containing ONLY those columns.
    :param copy: if true, we return a copy of the dataframe
    :return: subset of dataframe isolating the specified columns
    """
    res: pd.DataFrame = full_dataframe.loc[
                        :,
                        ~full_dataframe.columns.isin(special_columns) if remove_those_columns
                        else full_dataframe.columns.isin(special_columns)
                        ]

    if copy:
        return res.copy()
    return res


def isolate_this_column(
        full_dataframe: pd.DataFrame,
        isolate_this: str,
        remove_that_column: bool = True,
        copy: bool = True
) -> pd.DataFrame:
    """
    shortcut for either returning a dataframe containing everything BUT a certain column,
    or only returning the certain column
    :param full_dataframe: the full dataframe
    :param isolate_this: the singular column to isolate
    :param remove_that_column:
        if true, return a dataframe containing everything EXCEPT the specified column.
        if false, return a dataframe containing ONLY the specified column.
    :param copy: if true, return a copy of the dataframe. if false, return it without using .copy()
    :return:
        either a dataframe containing everything but the singular column,
         or a dataframe containing only the singular column.
    """
    res: pd.DataFrame = full_dataframe.loc[
                        :,
                        full_dataframe.columns != isolate_this if remove_that_column
                        else isolate_this
                        ]
    if copy:
        return res.copy()
    return res


def factual_counterfactual_e_splitter(
        full_dataframe: pd.DataFrame,
        counterfactual_columns: Iterable[str] = ("tcf", "ycf", "ite", "t0", "t1"),
        copy: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a dataframe, this splits it into a 'factual' dataframe
    (containing everything except the 'counterfactual columns')
    and a 'counterfactual' dataframe (containing *only* the counterfactual columns)

    :param full_dataframe: the full dataframe
    :param counterfactual_columns: the columns holding the counterfactual data,
    :param copy: do we want to copy the
    :return: a tuple of (dataframe with everything but counterfactuals, dataframe of only counterfactuals)
    """

    """
    return (
        full_dataframe.loc[:, ~full_dataframe.columns.isin(counterfactual_columns)],  # everything but counterfactuals
        full_dataframe.loc[:,  full_dataframe.columns.isin(counterfactual_columns)]   # only the counterfactuals
    )
    """

    # noinspection PyTypeChecker
    return tuple(
        isolate_these_columns(
            full_dataframe,
            counterfactual_columns,
            yeet_counterfactuals,
            copy
        )
        for yeet_counterfactuals in (True, False)
    )


def y_splitter(
        full_dataframe: pd.DataFrame,
        y_column: str,
        copy: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a dataframe, splits it into a dataframe containing everything BUT the 'y' column,
    and a dataframe containing ONLY the 'y' column.

    :param full_dataframe: the full dataframe to split
    :param y_column: the y column name
    :param copy: if we want copies of the dataframe or not
    :return: a tuple of (dataframe with everything but y, dataframe containing nothing but y)
    """
    """
    return (
        full_dataframe.loc[:, full_dataframe.columns != y_column],  # everything but y
        full_dataframe.loc[:, full_dataframe.columns == y_column]   # only y
    )
    """
    # noinspection PyTypeChecker
    return tuple(
        isolate_this_column(
            full_dataframe,
            y_column,
            yeet_y,
            copy
        ) for yeet_y in (True, False)
    )


def x_y_splitter(
        full_dataframe: pd.DataFrame,
        x_columns: Optional[Iterable[str]],
        y_column: str,
        copy: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe into a dataframe containing ONLY the x columns, and a dataframe containing ONLY the y_column
    :param full_dataframe: the full dataframe to split
    :param x_columns: iterable of the X column names. If not specified (or empty), calls y_splitter instead.
    :param y_column: the Y column name
    :param copy: if true, explicitly call .copy() for the return values.
    :return: tuple of [x columns dataframe, y column dataframe]
    """

    if iter_is_none(x_columns):
        return y_splitter(
            full_dataframe,
            y_column,
            copy
        )
    return (
        isolate_these_columns(
            full_dataframe=full_dataframe,
            special_columns=x_columns,
            remove_those_columns=False,
            copy=copy
        ),
        isolate_this_column(
            full_dataframe=full_dataframe,
            isolate_this=y_column,
            remove_that_column=False,
            copy=copy
        )
    )


"""

Stuff which is particularly relevant to the data science training/testing stuff

"""


def dataframes_to_numpy(
        dataframes: Iterable[pd.DataFrame]
) -> Tuple[np.ndarray, ...]:
    """
    converts the dataframes in the iterable to a tuple of ndarrays
    :param dataframes: an iterable of dataframes
    :return: same data but they're in a tuple and they're ndarrays instead
    """
    return tuple(
        df.copy().to_numpy()
        for df in dataframes
    )


def stratified_class_label_maker(
        the_dataframe: pd.DataFrame,
        columns_to_merge_for_labels: Iterable[str] = ("yf", "t", "e")
) -> np.ndarray:
    """
    creates an ndarray containing strings that are the stringified values
    of the columns_to_merge_for_labels for a particular row concatenated together with spaces between them.
    the intent is to give this to the stratification thing, to make it think that each combination of the
    values for these columns is a new class, so it will try to give an even distribution based on
    treatments and outcomes
    :param the_dataframe:
    :param columns_to_merge_for_labels:
    :return:
    """
    return isolate_these_columns(
        the_dataframe,
        columns_to_merge_for_labels,
        False,
        True
    ).astype(str).agg(" ".join, axis=1).values


def train_test_index_maker(
        the_dataframe: pd.DataFrame,
        test_size: float,
        columns_to_merge_for_labels: Iterable[str] = ("yf", "t", "e"),
        random_state: RNG = seed()
) -> Tuple[T_INDEX, T_INDEX]:
    """
    attempts to obtain the indices for the [training set, held-out validation set]
    :param the_dataframe:
    :param test_size:
    :param columns_to_merge_for_labels:
    :param random_state:
    :return:
    """
    train_indexes, test_indexes, _a, _b = train_test_split(
        the_dataframe.index,
        stratified_class_label_maker(
            the_dataframe,
            columns_to_merge_for_labels
        ),
        test_size=test_size,
        random_state=random_state
    )

    return train_indexes, test_indexes


class DatasetEnum(Enum):
    X_Y = auto()
    "represents 'only getting the X data (with no T)' (and y=yf)"
    X_T = auto()
    "represents x=x, y=t"
    FACTUAL = auto()
    "represents 'getting the x = (x + t factual) and y = y factual' data"
    COUNTERFACTUAL = auto()
    "represents 'getting the x = (x + t counterfactual) and y = y counterfactual' data"
    T0 = auto()
    "represents 'getting the x = (x + (t=0)) and y = y(t0)' data"
    T1 = auto()
    "represents 'getting the x = (x + (t=1)) and y = y(t1)' data"

    X_ITE = auto()
    "represents getting x=x, y=ITE"

    X_E = auto()
    "represents getting x=x, y=e"

    IPSW_X_TCF = auto()
    """
    Inverse propensity weighting: the INVERSE of whether an individual will be assigned to the treatment group;
    in other words, the likelihood of the individual NOT being treated.
    
    So this obtains x=x, y=tcf, attempting to predict the inverse of the individual's likelihood to get treated.
    """


class KFolder(Iterable[Tuple[T_INDEX, T_INDEX]]):
    """
    an iterator for the [train indices, test indices] for kfolds on a dataframe
    """

    def __init__(
        self,
        dataframe_to_kfold: pd.DataFrame,
        columns_to_stratify_on: Iterable[str] = ("yf", "t", "e"),
        random_state: RNG = seed(),
        n_splits: int = 10,
        shuffle: bool = False
    ):
        """
        constructor
        :param dataframe_to_kfold: the dataframe we're kfolding
        :param columns_to_stratify_on: names of the columns being used for stratification
        :param random_state: random state to use
        :param n_splits: number of folds
        :param shuffle: are we shuffling this? (set to false if using HalvingGridSearchCV)
        """
        self.shuffle = shuffle
        self.random_state = random_state if shuffle else None
        self.n_splits = n_splits

        self.class_labels = stratified_class_label_maker(
            dataframe_to_kfold,
            columns_to_stratify_on
        )

    def __iter__(self) -> Iterator[Tuple[T_INDEX, T_INDEX]]:
        """
        iterates through (train indices, test indices) for the folds
        :return: each fold's (train indices, test indices)
        """
        return (
            (train, test) for train, test in StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            ).split(
                np.zeros_like(self.class_labels),
                self.class_labels
            )
        )

    def outer_inner_iter(self) -> Iterator[
        Tuple[
            Tuple[T_INDEX, T_INDEX],
            Iterator[Tuple[T_INDEX, T_INDEX]]
        ]
    ]:
        """

        :return: Zip holding [(train fold, test fold), (iter(nested train, test))]
        """
        outer: Tuple[Tuple[T_INDEX, T_INDEX]] = tuple(self.__iter__())

        return zip(
            outer,
            (
                (
                    (tr, te) for tr, te in StratifiedKFold(
                        n_splits=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state
                    ).split(
                        np.zeros_like(o_train),
                        self.class_labels[o_train]
                    )

                ) for o_train, _ in outer
            )
        )



@dataclasses.dataclass(init=T, repr=True, frozen=True, eq=True)
class TrainTestXY:

    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_x:  pd.DataFrame
    test_y:  pd.DataFrame

    @classmethod
    def make(
            cls,
            x_y_df: Tuple[pd.DataFrame, pd.DataFrame],
            train_test_indices: Tuple[T_INDEX, T_INDEX],
    ) -> TrainTestXY:

        x, y = x_y_df
        train, test = train_test_indices

        return cls(
            train_x=x.loc[train],
            train_y=y.loc[train],
            test_x=x.loc[test],
            test_y=y.loc[test]
        )

@dataclasses.dataclass(init=True, repr=True, frozen=True, eq=True)
class DataframeManager:
    """
    Dataclass containing convenient abstractions for the train/test split stuff.
    """

    dataset_name: str
    "name of the dataset"

    _full_dataframe: pd.DataFrame
    "the full dataframe"

    _train_indices: T_INDEX
    "indices for the training set"
    _test_indices: T_INDEX
    "indices for the test set"

    test_proportion: float
    "obtains the test proportion (relative to the full dataset)"

    t_column: str
    "name of the 'treatment' column"

    t_cf_column: str
    "name of the counterfactual 'treatment' column"

    y_column: str
    "name of the 'y' column"

    ycf_column: Optional[str]
    "name of y counterfactual column (if present)"

    t0_column: Optional[str]
    "name of t=0 results column (if present)"

    t1_column: Optional[str]
    "name of t=1 results column (if present)"

    ite_column: Optional[str]
    "name of ITE column (if present)"

    e_column: str
    "name of E (is individual from experiment group or control group?) column"

    @classmethod
    def make(
            cls,
            dataset_name: str,
            the_df: pd.DataFrame,
            test_proportion: float,
            split_randomstate: RNG = seed(),
            columns_for_training_stratification: Iterable[str] = ("t", "e"),
            t_column: str = "t",
            t_cf_column: str = "tcf",
            y_column: str = "yf",
            e_column: str = "e",
            default_e_if_e_not_present: Any = 1,
            ycf_column: Optional[str] = None,
            t0_column: Optional[str] = None,
            t1_column: Optional[str] = None,
            ite_column: Optional[str] = None
    ) -> "DataframeManager":
        """

        :param dataset_name:
        :param the_df:
        :param test_proportion:
        :param split_randomstate:
        :param columns_for_training_stratification:
        :param t_column:
        :param t_cf_column:
        :param y_column:
        :param e_column:
        :param default_e_if_e_not_present: what to fill in the E column if there is no E column already in here.
        :param ycf_column:
        :param t0_column:
        :param t1_column:
        :param ite_column:
        :return:
        """

        df: pd.DataFrame = the_df.copy()

        if e_column not in the_df.columns:
            df[e_column] = default_e_if_e_not_present
            df = turn_01_columns_into_int(df)

        train, test = train_test_index_maker(
            df,
            test_size=test_proportion,
            columns_to_merge_for_labels=columns_for_training_stratification,
            random_state=split_randomstate
        )

        return cls(
            dataset_name=dataset_name,
            _full_dataframe=df,
            _train_indices=train,
            _test_indices=test,
            test_proportion=test_proportion,
            t_column=t_column,
            t_cf_column=t_cf_column,
            y_column=y_column,
            ycf_column=ycf_column,
            t0_column=t0_column,
            t1_column=t1_column,
            ite_column=ite_column,
            e_column=e_column
        )

    @cached_property
    def e1_dataframe(self) -> "DataframeManager":
        """:return: a DataframeManager for the subset of this dataframe where e=1"""
        if np.all(self._full_dataframe[self.e_column].to_numpy().flatten() == 1):
            return self

        return DataframeManager.make(
            dataset_name=self.dataset_name,
            the_df=self.full_dataframe.loc[self.full_dataframe[self.e_column] == 1],
            test_proportion=self.test_proportion,
            split_randomstate=seed(),
            columns_for_training_stratification=[self.t_column, self.y_column] if self.y_binary else [self.t_column],
            t_column=self.t_column,
            t_cf_column=self.t_cf_column,
            y_column=self.y_column,
            e_column=self.e_column,
            ycf_column=self.ycf_column,
            t0_column=self.t0_column,
            t1_column=self.t1_column,
            ite_column=self.ite_column
        )



    @property
    def full_dataframe(self) -> pd.DataFrame:
        """:return: a copy of the full dataframe"""
        return self._full_dataframe.copy()

    @cached_property
    def counterfactual_columns(self) -> Tuple[str, ...]:
        """:return: names of the non-factual columns"""
        return tuple(
            i for i in (
                self.t_cf_column,
                self.ycf_column,
                self.t0_column,
                self.t1_column,
                self.ite_column
            ) if i is not None
        )

    @property
    def t0_t1_ite_ycf_df_or_none(self) -> Optional[pd.DataFrame]:
        """
        If there are counterfactuals, returns a dataframe containing only the counterfactuals.
        otherwise returns nothing
        :return: dataframe of [ycf, t0, t1, ite] or None.
        """
        cf: Tuple[str, ...] = tuple(
            i for i in (
                self.ycf_column,
                self.t0_column,
                self.t1_column,
                self.ite_column
            ) if i is not None
        )
        if len(cf) == 0:
            return None
        else:
            return isolate_these_columns(
                self.full_dataframe,
                special_columns=cf,
                remove_those_columns=False
            )

    @property
    def y_binary(self) -> bool:
        """:return: true if Y has 2 (or fewer) discrete values"""
        return self._full_dataframe[self.y_column].nunique() < 3

    @cached_property
    def x_columns(self) -> Tuple[str, ...]:
        """:return: only the x column names"""
        return tuple(
            c_name for c_name in self._full_dataframe.columns.values
            if c_name not in chain(
                [self.t_column, self.y_column, self.e_column],
                self.counterfactual_columns
            )
        )

    @cached_property
    def xt_columns(self) -> Tuple[str, ...]:
        """:return: the x and factual t column names"""
        return tuple(
            chain_1(self.x_columns, self.t_column)
        )

    @cached_property
    def xt_counterfactual_columns(self) -> Tuple[str, ...]:
        """:return: the x and counterfactual t column names"""
        return tuple(
            chain_1(self.x_columns, self.t_cf_column)
        )

    @cached_property
    def full_factual_dataframe(self) -> pd.DataFrame:
        """:return: a copy of the full (factual) dataframe"""
        return isolate_these_columns(
            self._full_dataframe,
            special_columns=self.counterfactual_columns,
            remove_those_columns=True,
            copy=True
        )

    @property
    def train_indices(self) -> np.ndarray:
        """:return: copy of the training set indices"""
        return self._train_indices.copy()

    @property
    def test_indices(self) -> np.ndarray:
        """:return: copy of the test set indices"""
        return self._test_indices.copy()

    def get_from_indices(self, indices: T_INDEX, copy: bool = True) -> pd.DataFrame:
        """
        Obtain the specified rows from the dataframe
        :param indices: np.ndarray of the rows to select from the dataframe
        :param copy: if true, explicitly call 'copy()' on the result
        :return: the portion of the full dataframe with the specified rows
        """
        if copy:
            return self._full_dataframe.loc[indices, :].copy()
        else:
            return self._full_dataframe.loc[indices, :]

    def get_train_df(self, copy: bool = True) -> pd.DataFrame:
        """
        Obtain the training set portion of the dataframe
        :param copy: true if you want a copy of it, false if you want the raw one
        :return: the training set portion (or a copy of it)
        """
        return self.get_from_indices(self._train_indices, copy)

    @property
    def train_df(self) -> pd.DataFrame:
        """
        Obtain the training set portion of the dataframe
        :return: copy of training set
        """
        return self.get_train_df()

    def get_test_df(self, copy: bool = True) -> pd.DataFrame:
        """
        Obtain the test set portion of the dataframe
        :param copy: true if you want a copy of it, false if you want the raw one
        :return: the test set portion (or a copy of it)
        """
        return self.get_from_indices(self._test_indices, copy)

    @property
    def test_df(self) -> pd.DataFrame:
        """
        Obtain the test portion of the dataframe
        :return: copy of test set
        """
        return self.get_test_df()

    def get_df(
            self,
            train: Optional[bool],
            copy: bool = True
    ) -> pd.DataFrame:
        """
        easy method for returning the training set, the test set, or the full dataframe
        :param train: should we use the training set, the test set, or the full data?
            if None, use the full dataframe.
            if true, return data from the training set indices.
            if false, return data from the test set indices.
        :param copy: if true, return the dataframe as a copy instead of as a view of the raw values
        :return: either the full datframe, the train data, or the test data
        """
        df: pd.DataFrame

        if train is None:
            df = self._full_dataframe
            if copy:
                df = df.copy()
        elif train:
            df = self.get_train_df(copy)
        else:
            df = self.get_test_df(copy)
        return df

    def x_data(
            self,
            train: Optional[
                Union[
                    bool,
                    pd.DataFrame
                ]
            ],
            x_columns: Optional[
                Union[
                    Iterable[str],
                    DatasetEnum
                ]
            ] = None,
            copy: bool = True
    ) -> pd.DataFrame:
        """
        Returns a dataframe for the X data to give to a predictor.
        :param train: should we use the training set, the test set, or the full data?
            if a dataframe, just use that dataframe.
            if None, use the full dataframe.
            if true, return data from the training set indices.
            if false, return data from the test set indices.
        :param x_columns: Which X columns do we want?
            if a DatasetEnum is given, obtain the appropriate X columns for that.
                X_Y -> self.x_columns
                FACTUAL -> self.xt_columns
                COUNTERFACTUAL -> self.xt_counterfactual_columns
                T0 -> self.x_columns + (t=0)
                T1 -> self.x_columns + (t=1)
            If none or an empty iterable is given, get DatasetEnum.X results
            if an iterable is given, use those X columns.
        :param copy: do we want a copy of the dataset?
        :return: the appropriate dataframe
        """
        xdata: pd.DataFrame

        if train is None:
            xdata = self._full_dataframe
        elif isinstance(train, pd.DataFrame):
            xdata = train
        elif train:
            xdata = self.get_train_df(False)
        else:
            xdata = self.get_test_df(False)

        if (
                (x_columns is None) or
                (x_columns == DatasetEnum.X_Y) or
                (x_columns == DatasetEnum.X_T) or
                (x_columns == DatasetEnum.IPSW_X_TCF) or
                (x_columns == DatasetEnum.X_ITE) or
                (x_columns == DatasetEnum.X_E) or
                (hasattr(x_columns, "__iter__") and iter_is_none(x_columns))  # if x is iterable that contains nothing
        ):
            xdata = isolate_these_columns(
                full_dataframe=xdata,
                special_columns=self.x_columns,
                remove_those_columns=False,
                copy=copy
            )
        elif hasattr(x_columns, "__iter__"):
            # if x is iterable, it would have gone into the prior branch if it was empty/contained nones,
            # so we can safely assume that x_columns is not empty/is not full of nones
            xdata = isolate_these_columns(
                full_dataframe=xdata,
                special_columns=x_columns,
                remove_those_columns=False,
                copy=copy
            )

        elif x_columns == DatasetEnum.FACTUAL:
            xdata = isolate_these_columns(
                full_dataframe=xdata,
                special_columns=self.xt_columns,
                remove_those_columns=False,
                copy=copy
            )
        elif x_columns == DatasetEnum.COUNTERFACTUAL:
            xdata = isolate_these_columns(
                full_dataframe=xdata,
                special_columns=self.xt_counterfactual_columns,
                remove_those_columns=False,
                copy=copy
            )
        else:
            xdata = isolate_these_columns(
                full_dataframe=xdata,
                special_columns=self.x_columns,
                remove_those_columns=False,
                copy=True
            )
            if x_columns == DatasetEnum.T0:
                xdata[self.t_column] = 0
            else:
                xdata[self.t_column] = 1

        return xdata

    def get_e(
            self,
            train: Optional[
                Union[
                    bool,
                    pd.DataFrame
                ]
            ],
            copy: bool = True
    ) -> pd.DataFrame:
        """
        attempts to obtain e column (if it exists)
        :param train: should we use the training set, the test set, or the full data?
            if a dataframe, just use that dataframe.
            if None, use the full dataframe.
            if true, return data from the training set indices.
            if false, return data from the test set indices.
        :param copy: do we want a copy of the dataset?
        :return: the appropriate dataframe but only the e column of it.
        """
        return self.x_data(train, x_columns=[self.e_column], copy=copy)

    def x_y(
            self,
            train: Optional[
                Union[
                    bool,
                    pd.DataFrame
                ]
            ],
            x_columns: Optional[
                Union[
                    Iterable[str],
                    DatasetEnum
                ]
            ] = DatasetEnum.FACTUAL,
            y_column: Optional[
                Union[
                    str,
                    DatasetEnum
                ]
            ] = None,
            copy: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Obtains part of the dataset as a [x data, y column] tuple
        :param train: If true, return train split. If false, return test split. If None, use full dataframe.
        :param x_columns: The x columns to use.
            If an iterable of strings are given, we use those X columns
            If None or empty, treat it as DatasetEnum.X
            If DatasetEnum:
                X_Y -> self.x_columns
                FACTUAL -> self.xt_columns
                COUNTERFACTUAL -> self.xt_counterfactual_columns
                T0 -> self.x_columns + (t=0)
                T1 -> self.x_columns + (t=1)
        :param y_column: the Y column to use.
            If None, defers to x_columns parameter if x_columns is a DatasetEnum (else self.y_column).
            If a string, returns the column with that name.
            If a DatasetEnum:
                X_Y -> self.y_column
                FACTUAL -> self.y_column
                COUNTERFACTUAL -> self.y_cf_column
                T0 -> self.t0_column
                T1 -> self.t1_column
            RAISES A VALUEERROR IF YOU TRY TO OBTAIN A Y VALUE THAT DOESN'T EXIST!
        :param copy: if true, explicitly calls .copy() when obtaining the dataframes.
        :return: a tuple of [dataframe with the appropriate x values, dataframe of only the appropriate y values]
        """

        if y_column is None:

            if x_columns is not None and isinstance(x_columns, DatasetEnum):
                y_column = x_columns  # if x_columns is a DatasetEnum, y_column set to that value.
            else:
                y_column = DatasetEnum.FACTUAL  # otherwise, y_column defaults to FACTUAL

        if not isinstance(y_column, str):
            y_missing: bool = True
            if y_column == DatasetEnum.FACTUAL or y_column == DatasetEnum.X_Y:
                y_column = self.y_column
                y_missing = False
            elif y_column == DatasetEnum.X_T:
                y_missing = False
                y_column = self.t_column
            elif y_column == DatasetEnum.IPSW_X_TCF:
                y_missing = False
                y_column = self.t_cf_column
            elif y_column == DatasetEnum.X_E:
                y_missing = False
                y_column = self.e_column
            elif y_column == DatasetEnum.COUNTERFACTUAL:
                if self.ycf_column is not None:
                    y_missing = False
                    y_column = self.ycf_column
            elif y_column == DatasetEnum.T0:
                if self.t0_column is not None:
                    y_missing = False
                    y_column = self.t0_column
            elif y_column == DatasetEnum.T1:
                if self.t1_column is not None:
                    y_missing = False
                    y_column = self.t1_column
            elif y_column == DatasetEnum.X_ITE:
                if self.ite_column is not None:
                    y_missing = False
                    y_column = self.ite_column

            if y_missing:
                raise ValueError(
                    f"No appropriate Y column for y_column argument {y_column} could be found!"
                )

        assert isinstance(y_column, str)

        if train is None or not isinstance(train, pd.DataFrame):
            train = self.get_df(
                train,
                False
            )

        return (
            self.x_data(
                train,
                x_columns,
                copy
            ),
            isolate_this_column(
                full_dataframe=train,
                isolate_this=y_column,
                remove_that_column=False,
                copy=copy
            )
        )

    def get_kfold_indices(
            self,
            train: Optional[Union[bool, pd.DataFrame]] = True,
            random_state: RNG = seed(),
            class_columns: Iterable[str] = ("t", "e"),
            n_splits: int = 10,
            shuffle: bool = False
    ) -> KFolder:  # Iterable[Tuple[T_INDEX, T_INDEX]]:
        """
        handles getting the indices for the kfold stuff for the training set
        :param train: If true, use train set. If false, return test set. If None, use full dataframe.
        :param random_state:
        :param class_columns:
        :param n_splits:
        :param shuffle:
        :return:
        """
        if train is None:
            train: pd.DataFrame = self.full_dataframe
        elif isinstance(train, pd.DataFrame):
            train: pd.DataFrame = train
        elif train:
            train: pd.DataFrame = self.train_df
        else:
            train: pd.DataFrame = self.test_df
        return KFolder(
            train,
            columns_to_stratify_on=class_columns,
            random_state=random_state,
            n_splits=n_splits,
            shuffle=shuffle
        )

    def save_self(
            self
    ) -> bool:
        """
        Pickles this object, as `os.getcwd() + f"\\{self.dataset_name}\\{self.dataset_name} DataframeManager.pickle"`.
        :return: true if this could be specified, false otherwise.
        """

        rel_pos: str = f"\\{self.dataset_name}\\{self.dataset_name} DataframeManager.pickle"

        print(f"pickling self as {rel_pos}...")
        with open(f"{os.getcwd()}{rel_pos}", "wb") as the_file:
            pickle.dump(self, the_file)
            print("pickled!")
            success = True

        return success

    @staticmethod
    def load(
            filename: str
    ) -> "DataframeManager":
        """
        Loads a pickled DataframeManager instance
        :param filename: filename of the thing to load
        :return: the loaded previously-picked DataframeManager.
        """

        loaded: "DataframeManager"
        with open(filename, "rb") as the_file:
            loaded = pickle.load(the_file)

        assert isinstance(loaded, DataframeManager)

        return loaded


