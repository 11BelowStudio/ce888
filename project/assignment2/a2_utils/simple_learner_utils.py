
from __future__ import annotations

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn as sk
import numpy as np

from tempfile import mkdtemp
from shutil import rmtree

from typing import List, Tuple, Dict, Iterable, Iterator, TypeVar, Union, Optional, NoReturn, Any, Callable

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection._validation import NotFittedError
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config
from sklearn.metrics import r2_score, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV

from sklearn.metrics import r2_score, roc_auc_score

import assignment2.a2_utils.dataframe_utils as df_utils
from assignment2.a2_utils import metric_utils

from assignment2.a2_utils.metric_utils import *
from assignment2.a2_utils.seed_utils import *
from assignment2.a2_utils.misc_utils import *

from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from math import inf
import dataclasses
import os
import sys

from inspect import ismethod, getmembers

import traceback

import sklearn


from functools import cached_property

import pickle



import warnings
warnings.filterwarnings("ignore")

set_config(display="diagram")

R = TypeVar('R', bound=RegressorMixin)

C = TypeVar('C', bound=ClassifierMixin)

EST = Union[R, C]

PP = Union[R, C, "PPipeline"]



class PPipeline(Pipeline):
    """
    A wrapper for Pipeline that exposes a few extra convenience methods.
    """

    def __init__(self, steps: List[Tuple[str, Any]], *, memory=None, verbose: bool = False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def _fit(self, X, y=None, **fit_params_steps):
        """
        overrides _fit inner method, to temporarily create a cache for the transformers and such
        :param X:
        :param y:
        :param fit_params_steps:
        :return:
        """
        self.memory = mkdtemp()
        fit_res = super()._fit(X=X, y=y, **fit_params_steps)
        rmtree(self.memory)
        self.memory = None
        return fit_res

    @property
    def final_estimator(self) -> PP:
        return self._final_estimator

    @property
    def final_estimator_inner(self) -> EST:
        final_est: PP = self.final_estimator
        if isinstance(final_est, PPipeline):
            return final_est.final_estimator_inner
        elif isinstance(final_est, Pipeline):
            return final_est._final_estimator
        else:
            return final_est

    @cached_property
    def feature_importances_(self) -> np.ndarray:
        return get_importances_mdi(self.final_estimator)

    @cached_property
    def feature_importances_std_(self) -> np.ndarray:
        return get_importance_mdi_std(self.final_estimator)

    @property
    def estimator_type(self) -> str:
        return self._estimator_type

    def _get_estimator_from_type(self, expected_type: str) -> PP:
        if self.estimator_type == expected_type:
            return self.final_estimator
        else:
            raise TypeError(f"Final estimator is {self.estimator_type}, not {expected_type}!")

    @property
    def regressor(self) -> R:
        # noinspection PyProtectedMember
        return self._get_estimator_from_type(RegressorMixin._estimator_type)

    @property
    def classifier(self) -> C:
        # noinspection PyProtectedMember
        return self._get_estimator_from_type(ClassifierMixin._estimator_type)



def get_importances_mdi(predictor: PP) -> np.ndarray:
    """
    Obtains min decrease in impurity feature importances for a given predictor
    :param predictor: the predictor we want the MDI-based importances of
    :return: np.ndarray of the feature importances of the predictor.
    """
    try:
        return predictor.feature_importances_
    except AttributeError as e1:
        try:
            return predictor.coef_.flatten()
        except AttributeError as e2:
            try:
                return np.mean([get_importances_mdi(p) for p in predictor.estimators_], axis=0)
            except AttributeError as e3:
                # noinspection PyProtectedMember
                return get_importances_mdi(predictor._final_estimator)


def get_importance_mdi_std(predictor: PP) -> np.ndarray:
    """
    Obtains standard deviation of min decrease in impurity feature importances for a given predictor
    :param predictor: the predictor we want the MDI-based importances of
    :return: np.ndarray of the feature importances of the predictor.
    """
    try:
        return np.std([get_importances_mdi(p) for p in predictor.estimators_], axis=0)
    except AttributeError as e1:
        try:
            return np.std([get_importances_mdi(predictor)], axis=0)
        except AttributeError as e2:
            # noinspection PyProtectedMember
            return get_importance_mdi_std(predictor._final_estimator)


def get_permutation_importances(
        estimator: PP,
        x: pd.DataFrame,
        y: pd.DataFrame,
        n_repeats: int = 10,
        random_state: RNG = seed()
) -> pd.DataFrame:

    importance_results: Bunch = permutation_importance(
        estimator,
        x,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df: pd.DataFrame = pd.DataFrame()

    importance_df["features"] = x.columns.values

    importance_df["mean"] = importance_results.importances_mean
    importance_df["std"] = importance_results.importances_std
    importance_df["importances"] = importance_results.importances.tolist()

    importance_df.set_index("features")

    return importance_df


@dataclasses.dataclass(init=True, eq=True, repr=True, frozen=True)
class SimpleHalvingGridSearchResults:

    searched: HalvingGridSearchCV

    learner_name: str

    dataset_name: str

    test_indices: np.ndarray

    x_t_column_names: Tuple[str]

    predictions: pd.DataFrame

    feature_importances: pd.DataFrame

    train_score: float

    validation_fold_score: float

    yf_score: float


    # these will use a dummy value of 42 (because r2 score of 42 is impossible, and it's less ugly than a None)
    ycf_score: Optional[float]
    t0_score: Optional[float]
    t1_score: Optional[float]
    ite_score: Optional[float]

    # ITE/average treatment effect error metrics
    abs_ate: Optional[float]
    pehe: Optional[float]

    # ATT error metrics
    abs_att: Optional[float]
    policy_risk: Optional[float]


    # names of the special columns
    t_column: str
    "treatment"
    yf_column: str
    "y factual"
    ycf_column: str
    "y counterfactual"
    t0_column: str
    "y when t=0"
    t1_column: str
    "y when t=1"
    ite_column: str
    "individual treatment effects"

    e_data: np.ndarray
    "experimental/control group?"

    @property
    def has_counterfactual_score(self) -> bool:
        return self.ycf_score is not None

    @property
    def has_t0_t1_scores(self) -> bool:
        return self.t0_score is not None and self.t1_score is not None


    def pred_ate(self) -> float:
        """
        Uses ITE predictions to predict average effect on the treated
        :return: average effect on the treated
        """
        # noinspection PyTypeChecker
        return np.mean(self.predictions[self.ite_column])

    @property
    def has_pehe(self) -> bool:
        return self.pehe is not None

    @property
    def has_policy_risk(self) -> bool:
        return self.policy_risk is not None

    @property
    def summary_info(self) -> str:
        out_string: str = f"GridSearchResults summary {self.learner_name} {self.dataset_name}" \
                          f"\n\ttest score:\t{self.validation_fold_score}" \
                          f"\n\ttrain score:{self.train_score}" \
                          f"\n\t{self.yf_column} score:\t{self.yf_score}"
        if self.has_counterfactual_score:
            out_string = out_string + \
                         f"\n\tycf score:\t{self.ycf_score}"
        if self.has_t0_t1_scores:
            out_string = out_string + \
                         f"\n\tt0 score:\t{self.t0_score}" \
                         f"\n\tt1 score:\t{self.t1_score}" \
                         f"\n\tite score:\t{self.ite_score}"
        if self.has_pehe:
            out_string = out_string + \
                         f"\n\tabs ATE:\t{self.abs_ate}" \
                         f"\n\tPEHE:   \t{self.pehe}"
        if self.has_policy_risk:
            out_string = out_string + \
                         f"\n\tabs ATT:\t{self.abs_att}" \
                         f"\n\tp. risk:\t{self.policy_risk}"
        return out_string

    @property
    def info(self) -> str:
        out_string: str = self.summary_info

        out_string = out_string + "\n\tbest params:"

        out_string = out_string + "".join(
            f"\n\t\t{k} : {v}"
            for k, v in self.best_params_.items()
        )

        return out_string

    @property
    def best_params_(self) -> Dict[str, Any]:
        return self.searched.best_params_

    def __lt__(self, other: "SimpleHalvingGridSearchResults") -> bool:

        if self.validation_fold_score < other.validation_fold_score:
            return True
        elif self.validation_fold_score == other.validation_fold_score:

            if self.has_pehe and other.has_pehe:
                # lower PEHE is better, so, the one with a larger PEHE is considered 'less than'.
                if self.pehe > other.pehe:
                    return True
                elif self.pehe < other.pehe:
                    return False

            if self.has_policy_risk and other.has_policy_risk:
                # lower policy_risk is better, so, the one with a larger policy_risk is considered 'less than'.
                if self.policy_risk > other.policy_risk:
                    return True
                elif self.policy_risk < other.policy_risk:
                    return False

            if self.has_counterfactual_score and other.has_counterfactual_score:
                if self.ycf_score < other.ycf_score:
                    return True
                elif self.ycf_score > other.ycf_score:
                    return False

            if self.has_t0_t1_scores and other.has_t0_t1_scores:
                # basically the r2 scores for t1 and t0 are shifted down to
                # have an upper limit of -1, then the products of the
                # shifted r2 scores are found.
                # higher product = worse r2 scores: counted as 'less than'
                # returns true if this object's combined r2 is worse than other.
                if (
                        (self.t0_score - 2) * (self.t1_score - 2)
                ) > (
                        (other.t0_score - 2) * (other.t1_score - 2)
                ):
                    return True
                else:
                    return False
            # otherwise compare the factual r2 scores
            return self.yf_score < other.yf_score
        # this is not less than the other thing
        return False

    @property
    def best_estimator_(self) -> PPipeline:
        return self.searched.best_estimator_

    @property
    def clone_best_final_estimator(self) -> EST:
        return sk.base.clone(self.best_estimator_.final_estimator)

    def plot_feature_importances(
            self,
            ax: plt.Axes,
            permutation: bool = True
    ) -> plt.Axes:
        """
        Plots feature importances (using matplotlib stuff)
        :param ax: the matplotlib axes we're trying to plot feature importances on
        :param permutation: if true, we plot the feature importances calculated from a permutation test on the test set. If false, we work out mean decrease in impurity
        :return: the axes with feature importances plotted on them
        """

        importance: np.ndarray = np.zeros_like(self.feature_importances.index)
        error: np.ndarray = np.zeros_like(self.feature_importances.index)

        y_label: str = "PLACEHOLDER"
        plot_title: str = "PLACEHOLDER"

        if permutation:

            importance = self.feature_importances["mean"].values
            error = self.feature_importances["std"].values

            plot_title = f"{self.dataset_name} feature importances using permutation for {self.learner_name}"

            y_label = "Mean accuracy decrease"

        else:

            importance = self.best_estimator_.feature_importances_
            error = self.best_estimator_.feature_importances_std_

            plot_title = f"{self.dataset_name} feature importances using Mean Decrease in Impurity (no permutation) for {self.learner_name}"

            y_label = "Mean decrease in impurity"

        bar: matplotlib.container.BarContainer = ax.bar(
            self.x_t_column_names,
            importance,
            yerr=error
        )

        ax.bar_label(
            bar,
            labels=[
                f'{val:.3f}\n±{err:.3f}'
                for val, err in zip(importance, error)
            ]
        )

        ax.axhline(0, color='grey', linewidth=0.8)

        ax.grid(visible=True, which="both", axis="both")

        ax.set_title(plot_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel("feature names")

        return ax

    def importance_plotter(self) -> plt.Figure:

        faa: Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]] = plt.subplots(
            nrows=2,
            ncols=1,
            squeeze=True,
            figsize=(16, 18),
            tight_layout=True
        )

        fig: plt.Figure = faa[0]
        axs: Tuple[plt.Axes, plt.Axes] = faa[1]

        for ax, perm in zip(axs, [False, True]):

            self.plot_feature_importances(ax, perm)

        fig.suptitle(f"{self.dataset_name} feature importance graphs for {self.learner_name}")

        rel_filename: str = f"\\{self.dataset_name}\\{self.dataset_name} {self.learner_name} feature importances.pdf"
        print(f"Exporting feature importance graph to {rel_filename}")
        fig.savefig(
            fname=f"{os.getcwd()}{rel_filename}"
        )

        return fig

    def save_me(self) -> NoReturn:

        rel_folder: str = f"\\{self.dataset_name}\\"

        results_go_here_rel: str = f"{rel_folder}{self.dataset_name} {self.learner_name} results.pickle"

        estimator_goes_here_rel: str = f"{rel_folder}{self.dataset_name} {self.learner_name} estimator.pickle"

        print(f"Pickling results to: {results_go_here_rel}")

        with open(f"{os.getcwd()}{results_go_here_rel}", "wb") as resultsPickle:
            pickle.dump(self, resultsPickle)
            print("pickled results!")

        print(f"Pickling simple estimator to {estimator_goes_here_rel}")

        with open(f"{os.getcwd()}{estimator_goes_here_rel}", "wb") as estPickle:
            pickle.dump(self.best_estimator_, estPickle)
            print("pickled estimator!")

    @classmethod
    def load_me(cls, save_location: str) -> "SimpleHalvingGridSearchResults":

        with open(save_location, "rb") as load_pickle:
            return pickle.load(load_pickle, fix_imports=True)

    @classmethod
    def make_dfm(
            cls,
            searcher: HalvingGridSearchCV,
            df_m: df_utils.DataframeManager,
            xy: df_utils.DatasetEnum,
            learner_name: str,
            scorer_to_use: Literal["roc_auc", "r2"] = "r2",
            y_name: Optional[str] = None,
            t_name: Optional[str] = None,
            ycf_name: Optional[str] = None,
            t0_name: Optional[str] = None,
            t1_name: Optional[str] = None,
            ite_name: Optional[str] = None,
            **kwargs
    ) -> "SimpleHalvingGridSearchResults":

        all_x, all_yf = df_m.x_y(
            train=None,
            x_columns=xy
        )

        if y_name is None:
            y_name = df_m.y_column

        if t_name is None:
            t_name = df_m.t_column

        if ycf_name is None:
            ycf_name = df_m.ycf_column

        if t0_name is None:
            t0_name = df_m.t0_column

        if t1_name is None:
            t1_name = df_m.t1_column

        if ite_name is None:
            ite_name = df_m.ite_column

        return cls.make(
            searcher=searcher,
            dataset_name=df_m.dataset_name,
            learner_name=learner_name,
            all_x_data=all_x,
            factual_y_data=all_yf,
            scorer=r2_score if scorer_to_use == "r2" else roc_auc_score,
            test_indices=df_m.test_indices,
            e_data=df_m.get_e(train=None)[df_m.e_column].to_numpy(),
            true_t0_t1_ite_ycf=df_m.t0_t1_ite_ycf_df_or_none,
            t_column=df_m.t_column,
            yf_column=df_m.y_column,
            ycf_column=x_else(df_m.ycf_column, "ycf"),
            t0_column=x_else(df_m.t0_column, "t0"),
            t1_column=x_else(df_m.t1_column, "t1"),
            ite_column=x_else(df_m.ite_column, "ite")
        )

    @classmethod
    def make(
            cls,
            searcher: HalvingGridSearchCV,
            dataset_name: str,
            learner_name: str,
            all_x_data: pd.DataFrame,
            factual_y_data: pd.DataFrame,
            scorer: Callable[[np.ndarray, np.ndarray], float],
            test_indices: df_utils.T_INDEX,
            e_data: Optional[np.ndarray] = None,
            true_t0_t1_ite_ycf: Optional[pd.DataFrame] = None,
            t_column: str = "t",
            yf_column: str = "yf",
            ycf_column: str = "ycf",
            t0_column: str = "t0",
            t1_column: str = "t1",
            ite_column: str = "ite",
    ) -> "SimpleHalvingGridSearchResults":

        has_t: bool = t_column in all_x_data  # here for the things where it attempts to predict T.

        this_x: pd.DataFrame = all_x_data.copy()

        this_x_y: pd.DataFrame = this_x.copy()

        train_score: float = searcher.best_score_

        validation_score: float = searcher.score(
            this_x.loc[test_indices],
            factual_y_data.loc[test_indices]
        )

        x_t_names: Tuple[str] = tuple(this_x.columns.values)

        importance_df: pd.DataFrame = get_permutation_importances(
            searcher.best_estimator_,
            this_x.loc[test_indices],
            factual_y_data.loc[test_indices],
            n_repeats=10,
            random_state=seed()
        )

        this_x_y[yf_column] = searcher.predict(
            this_x.to_numpy()
        )

        factual_score: float = scorer(
            factual_y_data.to_numpy(),
            this_x_y[yf_column].to_numpy()
        )

        # dummy values of 42 because  r2 score of 42 is impossible.
        ycf_score: Optional[float] = None
        t0_score: Optional[float] = None
        t1_score: Optional[float] = None
        ite_score: Optional[float] = None

        _abs_ate: Optional[float] = None
        _pehe: Optional[float] = None

        _abs_att: Optional[float] = None

        _policy_risk: Optional[float] = None

        if has_t:
            old_t = this_x[t_column].to_numpy(copy=True)

            this_x[t_column] = np.choose(this_x[t_column].values, [1, 0])

            this_x_y[ycf_column] = searcher.predict(
                this_x.to_numpy()
            )

            this_x[t_column] = 0

            this_x_y[t0_column] = searcher.predict(this_x.to_numpy())

            this_x[t_column] = 1

            this_x_y[t1_column] = searcher.predict(this_x.to_numpy())

            this_x[t_column] = old_t

            this_x_y[ite_column] = this_x_y[t1_column] - this_x_y[t0_column]

            if true_t0_t1_ite_ycf is not None:
                ycf_score = scorer(
                    true_t0_t1_ite_ycf[ycf_column].to_numpy(),
                    this_x_y[ycf_column].to_numpy()
                )
                t0_score = scorer(
                    true_t0_t1_ite_ycf[t0_column].to_numpy(),
                    this_x_y[t0_column].to_numpy()
                )
                t1_score = scorer(
                    true_t0_t1_ite_ycf[t1_column].to_numpy(),
                    this_x_y[t1_column].to_numpy()
                )

                real_ite = true_t0_t1_ite_ycf[ite_column].to_numpy()
                pred_ite = this_x_y[ite_column].to_numpy()

                ite_score = scorer(
                    real_ite,
                    pred_ite
                )

                _abs_ate = abs_ate(
                    real_ite,
                    pred_ite
                )
                _pehe = pehe(
                    real_ite,
                    pred_ite
                )

            if e_data is None:
                e_data = np.ones_like(this_x[t_column].to_numpy())

            pred_ite: np.ndarray = this_x_y[ite_column].to_numpy()
            _yf: np.ndarray = this_x_y[yf_column].to_numpy()
            _t: np.ndarray = this_x_y[t_column].to_numpy()

            _abs_att = abs_att(
                pred_ite,
                _yf,
                _t,
                e_data
            )

            _policy_risk = policy_risk(
                pred_ite,
                _yf,
                _t,
                e_data
            )

        return SimpleHalvingGridSearchResults(
            searched=searcher,
            dataset_name=dataset_name,
            learner_name=learner_name,
            test_indices=test_indices,
            x_t_column_names=x_t_names,
            predictions=this_x_y,
            feature_importances=importance_df,
            train_score=train_score,
            validation_fold_score=validation_score,
            yf_score=factual_score,
            ycf_score=ycf_score,
            t0_score=t0_score,
            t1_score=t1_score,
            ite_score=ite_score,
            abs_ate=_abs_ate,
            pehe=_pehe,
            abs_att=_abs_att,
            policy_risk=_policy_risk,
            t_column=t_column,
            yf_column=yf_column,
            ycf_column=ycf_column,
            t0_column=t0_column,
            t1_column=t1_column,
            ite_column=ite_column,
            e_data=e_data
        )


def standalone_feature_importance_plotter(
        predictor: PP,
        x_data: pd.DataFrame,
        y_data: pd.DataFrame,
        predictor_name: str,
        dataset_name: str,
        n_repeats: int = 10,
        random_state: RNG = seed(),
) -> plt.Figure:

    feature_importances: pd.DataFrame = get_permutation_importances(
        predictor,
        x_data,
        y_data,
        n_repeats=n_repeats,
        random_state=random_state
    )

    x_t_names: Tuple[str] = tuple(x_data.columns.values)

    faa: Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]] = plt.subplots(
        nrows=2,
        ncols=1,
        squeeze=True,
        figsize=(16, 18),
        tight_layout=True
    )

    fig: plt.Figure = faa[0]
    axs: Tuple[plt.Axes, plt.Axes] = faa[1]

    for ax, perm in zip(axs, [False, True]):

        importance: np.ndarray = np.zeros_like(feature_importances.index)
        error: np.ndarray = np.zeros_like(feature_importances.index)

        y_label: str = "PLACEHOLDER"
        plot_title: str = "PLACEHOLDER"

        if perm:
            importance = feature_importances["mean"].values
            error = feature_importances["std"].values

            plot_title = f"{dataset_name} feature importances using permutation for {predictor_name}"

            y_label = "Mean accuracy decrease"
        else:
            importance = get_importances_mdi(predictor)
            error = get_importance_mdi_std(predictor)

            plot_title = f"{dataset_name} feature importances using Mean Decrease in Impurity (no permutation) for {predictor_name}"

            y_label = "Mean decrease in impurity"

        bar: matplotlib.container.BarContainer = ax.bar(
            x_t_names,
            importance,
            yerr=error
        )

        ax.bar_label(
            bar,
            labels=[
                f'{val:.3f}\n±{err:.3f}'
                for val, err in zip(importance, error)
            ]
        )

        ax.axhline(0, color='grey', linewidth=0.8)

        ax.grid(visible=True, which="both", axis="both")

        ax.set_title(plot_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel("feature names")

    fig.suptitle(f"{dataset_name} feature importance graphs for {predictor_name }")

    rel_filename: str = f"\\{dataset_name}\\{dataset_name} {predictor_name} feature importances.pdf"
    print(f"Exporting feature importance graph to {rel_filename}")
    fig.savefig(
        fname=f"{os.getcwd()}{rel_filename}"
    )

    return fig


def fallback_w_h_calc(split_this: int) -> Tuple[int, int]:
    h = np.floor(np.sqrt(split_this))
    # get something relatively close to the square
    w = split_this//h

    i_w = w < h
    if w * h < split_this:
        if i_w:
            w += 1
        else:
            h += 1
        i_w = not i_w
    return w, h

def w_h_finder(value_to_split: int) -> Tuple[int, int]:
    w: int = 1
    h: int = value_to_split
    if np.sqrt(value_to_split) % 1 == 0:  # if the filter count is a square number
        w = h = np.floor(np.sqrt(value_to_split))  # nice square layout for plots
    elif value_to_split & 1 == 1:  # if odd number of filter items
        w, h = fallback_w_h_calc(value_to_split)
    else:
        # if filter count is even, we work out which binary rectangle works best basically
        a: int = 1
        b: int = value_to_split
        while b > a and b & 1 == 0:
            b //= 2
            a *= 2
        if max(a, b) > min(a, b) * 2:
            w, h = fallback_w_h_calc(value_to_split)
        else:
            w = max(a, b)
            h = min(a, b)
    return int(w), int(h)


def factor_finder(split_this: int) -> Tuple[int, int]:
    """
    finds a usable min_res, factor for use in the halvinggridsearch. trying to maximize factor.
    :param split_this:
    :return:
    """

    # find the factor with most splits for split_this.
    # for i in range(1, sqrt)

    #floor_sqrt: int = np.floor(np.sqrt(split_this))

    factors: Dict[int, Tuple[int, int]] = dict(
        (i, factor_tester(split_this, i)) for i in range(2, int(np.ceil(np.sqrt(split_this))) + 1)
    )

    best_so_far: Tuple[int, int, int] = (-1, -1, 0)

    for fac, v in factors.items():
        if fac > best_so_far[0]:
            best_so_far = (fac, v[0], v[1])

    if best_so_far[2] == 0:
        return w_h_finder(split_this)
    else:
        return best_so_far[1], best_so_far[0]


def factor_tester(split_this: int, factor: int, tally: int = 0) -> Tuple[int, int]:
    # TODO: how many times can split_this be int divided by i? result of doing these chained divisions?
    # return div count of 0 if can't split.

    splitted = split_this / factor
    if (splitted % 1) == 0:
        tally += 1
        return factor_tester(split_this//factor, factor, tally)
    else:
        return split_this, tally


"""
def simple_halving_grid_searcher(
        estimator: EST,
        param_grid: Dict[str, List[Any]],
        train_data: np.ndarray,
        train_targets: np.ndarray,
        k_folds: Union[KFold, Iterable[Tuple[np.ndarray, np.ndarray]]] = KFold(n_splits=10, shuffle=False),
        sample_weights: Optional[np.ndarray] = None,
        resource: str = "n_samples"
) -> HalvingGridSearchCV:

    pipe: PPipeline = PPipeline(steps=[
        ("scaler", QuantileTransformer(output_distribution="normal")),
        ("imputer",KNNImputer(add_indicator=False, weights="distance")),
        ("estimator", estimator)
    ])

    n_splits: int = k_folds.get_n_splits() if isinstance(k_folds, skl.model_selection.BaseCrossValidator) else len(k_folds)

    n_max_resources: int = train_targets.size if resource == "n_samples" else pipe.get_params(deep=True)[resource]

    min_res, factor = factor_finder(n_max_resources) #w_h_finder(n_max_resources)

    print(f"max: {n_max_resources}, min: {min_res}, factor: {factor}")

    h_grid_search: HalvingGridSearchCV = HalvingGridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        factor=factor,
        cv=k_folds,
        scoring=make_scorer(r2_score),
        refit=True,
        verbose=1,
        n_jobs=-1,
        aggressive_elimination=True,
        error_score=-1000000000000,
        # I wanted to make this error score negative infinity, however, doing so caused a lot of
        # particularly unsightly warning messages to appear.

        # So, to save everyone involved from having to look at at a buttload of them with a buttload of numbers in them,
        # I'm just setting this to an incredibly low finite number which should be rather hard to reach.
        # And if this score (or an even lower score) somehow is reached legitimately, chances are that
        # the legitimate score being lower than the error score will be the least of one's concerns.
        resource=resource,
        max_resources= n_max_resources,
        min_resources= min_res #n_max_resources//4
    )

    if class_weights is not None:

        h_grid_search.fit(
            train_data, train_targets, sample_weight=class_weights
        )
    else:
        h_grid_search.fit(
            train_data, train_targets
        )

    return h_grid_search
"""


def r2_score_id() -> Literal["r2"]:
    """:return: 'r2', the string identifier for the r2 scoring function"""
    return "r2"


def roc_auc_id() -> Literal["roc_auc"]:
    """:return: 'roc_auc', the string identifier for the 'reciever operating characteristic area under curve' scoring function"""
    return "roc_auc"


@dataclasses.dataclass(init=True, frozen=True)
class IpswWrapper:
    """
    convenience object for the IPSW calculation things
    """

    ipsw_clf: ClassifierMixin

    dataset_name: str

    @classmethod
    def make(
            cls,
            ipsw_clf: Union[ClassifierMixin, PPipeline],
            dataset_name: str
    ) -> "IpswWrapper":
        # noinspection PyTypeChecker
        return cls(
            ipsw_clf = sklearn.base.clone(ipsw_clf),
            dataset_name=dataset_name
        )

    def ipsw(self, x, t) -> np.ndarray:
        return metric_utils.get_ps_weights(
            sklearn.base.clone(self.ipsw_clf), x, t
        )

    def save_me(self) -> NoReturn:

        path: str = f"{os.getcwd()}\\{self.dataset_name}\\{self.dataset_name} IPSW wrapper.pickle"

        with open(path, "wb") as save_here:
            pickle.dump(self, save_here, fix_imports=True)

    @classmethod
    def load_me(cls, save_location: str) -> "IpswWrapper":
        with open(save_location, "rb") as load_pickle:
            return pickle.load(load_pickle, fix_imports=True)





def simple_halving_grid_searcher(
    estimator: EST,
    param_grid: Dict[str, List[Any]],
    df_m: df_utils.DataframeManager,
    df_selection: df_utils.DatasetEnum,
    learner_name: str,
    kfold_splits: int = 10,
    scorer_to_use: Literal["roc_auc", "r2"] = "r2",
    stratify_on: Iterable[str] = tuple("t"),
    ipsw_calc: Optional[IpswWrapper] = None,
    nested_rng_generator: RNG = None,
    resource: str = "n_samples",
    resource_param_values: Optional[Iterable[int]] = None,
    **kwargs
) -> List[SimpleHalvingGridSearchResults]:
    """

    :param estimator:
    :param param_grid:
    :param df_m:
    :param df_selection:
    :param learner_name:
    :param kfold_splits:
    :param scorer_to_use: the scorer to use. either "r2" or "roc_auc"
    :param stratify_on:
    :param ipsw_calc:
    :param nested_rng_generator:
    :param resource: the resource used for the halving grid search.
    :param resource_param_values:
        If we're using an estimator parameter as a resource,
        this holds the values for that parameter.
    :param true_t0_t1_ite_ycf:
    :return:
    """

    if nested_rng_generator is None:
        nested_rng_generator = rng_state()

    resource_is_samples: bool = resource == "n_samples"

    if resource_is_samples:
        resource_param_values = [len(df_m.train_indices)]  # size of training set

    results: List[SimpleHalvingGridSearchResults] = []



    child_splits: int = max(1, kfold_splits-1)
    child_kf: Union[KFold, Iterable[Tuple[np.ndarray, np.ndarray]]] = KFold(n_splits=child_splits, shuffle=False)

    the_splits: Iterable[Tuple[df_utils.T_INDEX, df_utils.T_INDEX]] = df_m.get_kfold_indices(
        train=True,
        random_state=nested_rng_generator,
        class_columns=stratify_on,
        n_splits=kfold_splits
    )

    train_data, train_labels = df_m.x_y(True, x_columns=df_selection)
    #test_data, test_labels = df_m.x_y(False, x_columns=df_selection)

    for i, max_res in enumerate(resource_param_values, 1):

        print(f"-- 10-fold attempt {i}/{len(resource_param_values)} start --")

        pipe: PPipeline = PPipeline(steps=[
            ("scaler", QuantileTransformer(output_distribution="normal")),
            ("imputer", KNNImputer(add_indicator=False, weights="distance")),
            ("estimator", estimator)
        ])

        if not resource_is_samples:
            pipe.set_params(**{resource: max_res})

        min_res, factor = factor_finder(max_res)  # w_h_finder(n_max_resources)
        print(f"max: {max_res}, min: {min_res}, factor: {factor}")

        current_search: HalvingGridSearchCV = HalvingGridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            factor=factor,
            cv=the_splits,
            scoring=scorer_to_use,
            refit=True,
            verbose=1,
            n_jobs=-1,
            aggressive_elimination=True,
            error_score=-1000000000000,
            # I wanted to make this error score negative infinity, however, doing so caused a lot of
            # particularly unsightly warning messages to appear.

            # So, to save everyone involved from having to look at at a buttload of them with a buttload of numbers in them,
            # I'm just setting this to an incredibly low finite number which should be rather hard to reach.
            # And if this score (or an even lower score) somehow is reached legitimately, chances are that
            # the legitimate score being lower than the error score will be the least of one's concerns.
            resource=resource,
            max_resources= max_res,
            min_resources= min_res #n_max_resources//4
        )

        if ipsw_calc is not None:

            _x, _t = df_utils.x_y_splitter(
                train_data,
                x_columns=df_m.x_columns,
                y_column=df_m.y_column
            )

            current_search.fit(
                train_data, train_labels, sample_weight=ipsw_calc.ipsw(
                    _x, _t
                )
            )
        else:
            current_search.fit(
                train_data, train_labels
            )

        res: SimpleHalvingGridSearchResults = SimpleHalvingGridSearchResults.make_dfm(
            searcher=current_search,
            df_m=df_m,
            xy=df_selection,
            learner_name=learner_name,
            scorer_to_use=scorer_to_use,
            **kwargs
        )

        results.append(res)

        print(f"best from this iteration: \n{res.summary_info}")

        print(f"\n--- {i}/{len(resource_param_values)} END ---")

    return results

"""

    for i, (train_indices, test_indices) in enumerate(the_splits, 1):
        print(f"-- {i}/{kfold_splits} start --")

        pipe: PPipeline = PPipeline(steps=[
            ("scaler", QuantileTransformer(output_distribution="normal")),
            ("imputer", KNNImputer(add_indicator=False, weights="distance")),
            ("estimator", estimator)
        ])

        n_max_resources: int = len(train_indices) if resource == "n_samples" else pipe.get_params(deep=True)[resource]
        min_res, factor = factor_finder(n_max_resources)  # w_h_finder(n_max_resources)
        print(f"max: {n_max_resources}, min: {min_res}, factor: {factor}")


        current_search: HalvingGridSearchCV = HalvingGridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            factor=factor,
            cv = the_splits
        )


        current_search: HalvingGridSearchCV = simple_halving_grid_searcher(
            estimator=estimator,
            param_grid=param_grid,
            train_data=train_indices,
            train_targets=
        )




        try:


            if sample_weights is not None:
                child_kf = [
                    j for j in StratifiedKFold(
                        n_splits=child_splits,
                        shuffle=False
                    ).split(
                        X = np.zeros_like(train_indices),
                        y = sample_weights[train_indices]
                    )
                ]

            if sample_weights is not None:

                train_classes: np.ndarray = np.take(sample_weights, train_indices)

                train_classes = train_classes / np.sum(train_classes)

                test_classes: np.ndarray = np.take(sample_weights, test_indices)
                test_classes = test_classes / np.sum(test_classes)

                current_search: HalvingGridSearchCV = simple_halving_grid_searcher(
                    estimator,
                    param_grid,
                    learn_x_t_data.values[train_indices],
                    learn_y_targets.values[train_indices],
                    child_kf,
                    class_weights = train_classes,
                    resource=resource
                )

                results.append(
                    SimpleHalvingGridSearchResults.make(
                        current_search,
                        regressor_name,
                        learn_x_t_data,
                        learn_y_targets,
                        test_indices,
                        true_t0_t1_ite_ycf
                    )
                )

                current_score: float = current_search.score(
                    learn_x_t_data.values[test_indices],
                    learn_y_targets.values[test_indices]
                )

                h_grid_search_dicts[current_search] = current_score

            else:


                current_search: HalvingGridSearchCV = simple_halving_grid_searcher(
                    estimator,
                    param_grid,
                    learn_x_t_data.values[train_indices],
                    learn_y_targets.values[train_indices],
                    child_kf,
                    resource=resource
                )

                current_score: float = current_search.score(
                    learn_x_t_data.values[test_indices],
                    learn_y_targets.values[test_indices]
                )

                results.append(
                    SimpleHalvingGridSearchResults.make(
                        current_search,
                        regressor_name,
                        learn_x_t_data,
                        learn_y_targets,
                        test_indices,
                        true_t0_t1_ite_ycf
                    )
                )

                h_grid_search_dicts[current_search] = current_score

            print(f"-- {i}/{kfold_splits} done. Score: {current_score} --")

        except NotFittedError as e:
            print("oh no! there was a not fitted error!", sys.stderr)
            print(e, sys.stderr)
            print(traceback.format_exc(), sys.stderr)




    return results

"""


