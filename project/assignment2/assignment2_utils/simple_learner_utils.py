
from __future__ import annotations

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn as skl
import numpy as np

from tempfile import mkdtemp
from shutil import rmtree

from typing import List, Tuple, Dict, Iterable, Iterator, TypeVar, Union, Optional, NoReturn, Any

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

import dataframe_utils as df_utils

from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from math import inf
import dataclasses
import os
import sys

import traceback


from functools import cached_property

import pickle

from metric_utils import *
from seed_utils import *
from misc_utils import *

import warnings
warnings.filterwarnings("ignore")

set_config(display="diagram")

R = TypeVar('R', bound=RegressorMixin)

C = TypeVar('C', bound=ClassifierMixin)

EST = Union[R, C]

PP = Union[R, C, "PPipeline"]

from dataframe_utils import T_INDEX


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
        return PPipeline.get_importances_mdi(self.final_estimator)

    @cached_property
    def feature_importances_std_(self) -> np.ndarray:
        return PPipeline.get_importance_mdi_std(self.final_estimator)

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

    @staticmethod
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
                    return np.mean([PPipeline.get_importances_mdi(p) for p in predictor.estimators_],axis=0)
                except AttributeError as e3:
                    # noinspection PyProtectedMember
                    return PPipeline.get_importances_mdi(predictor._final_estimator)

    @staticmethod
    def get_importance_mdi_std(predictor: PP) -> np.ndarray:
        """
        Obtains standard deviation of min decrease in impurity feature importances for a given predictor
        :param predictor: the predictor we want the MDI-based importances of
        :return: np.ndarray of the feature importances of the predictor.
        """
        try:
            return np.std([PPipeline.get_importances_mdi(p) for p in predictor.estimators_], axis=0)
        except AttributeError as e1:
            try:
                return np.std([PPipeline.get_importances_mdi(predictor)], axis=0)
            except AttributeError as e2:
                return PPipeline.get_importance_mdi_std(predictor._final_estimator)



@dataclasses.dataclass(init=True, eq=True, repr=True, frozen=True)
class SimpleHalvingGridSearchResults:

    gridSearch: HalvingGridSearchCV

    learner_type: str

    dataset_name: str

    test_indices: np.ndarray

    x_t_column_names: Tuple[str]

    predictions: pd.DataFrame

    feature_importances: pd.DataFrame

    gridsearch_train_r2_score: float

    validation_fold_r2_score: float

    yf_r2_score: float


    # these will use a dummy value of 42 (because r2 score of 42 is impossible, and it's less ugly than a None)
    ycf_r2_score: Optional[float]
    t0_r2_score: Optional[float]
    t1_r2_score: Optional[float]

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
    e_column: str
    "experimental/control group?"

    @property
    def has_counterfactual_score(self) -> bool:
        return self.ycf_r2_score is not None

    @property
    def has_t0_t1_scores(self) -> bool:
        return self.t0_r2_score is not None and self.t1_r2_score is not None


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
        out_string: str = f"GridSearchResults summary {self.learner_type} {self.dataset_name}\n" \
                          f"\n\ttest r2: {self.validation_fold_r2_score}" \
                          f"\n\ttrain r2:{self.gridsearch_train_r2_score}" \
                          f"\n\tyf r2:   {self.yf_r2_score}"
        if self.has_pehe:
            out_string = out_string + \
                         f"\n\tabs ATE: {self.abs_ate}" \
                         f"\n\tPEHE:    {self.pehe}"
        if self.has_policy_risk:
            out_string = out_string + \
                         f"\n\tabs ATT: {self.abs_att}" \
                         f"\n\tp. risk: {self.policy_risk}"
        return out_string

    @property
    def info(self) -> str:
        out_string: str = f"GridSearchResults: {self.learner_type} {self.dataset_name}\n" \
                          f"\testimator: {self.best_estimator_}\n" \
                          f"\ttest r2: {self.validation_fold_r2_score}\n" \
                          f"\ttrain r2:{self.gridsearch_train_r2_score}\n" \
                          f"\tyf r2:   {self.yf_r2_score}"
        if self.has_counterfactual_score:
            out_string = out_string + \
                         f"\n\tycf r2:  {self.ycf_r2_score}"
        if self.has_t0_t1_scores:
            out_string = out_string + \
                         f"\n\tt0 r2:   {self.t0_r2_score}" \
                         f"\n\tt1 r2:   {self.t1_r2_score}"
        if self.has_pehe:
            out_string = out_string + \
                         f"\n\tabs ATE: {self.abs_ate}" \
                         f"\n\tPEHE:    {self.pehe}"
        if self.has_policy_risk:
            out_string = out_string + \
                         f"\n\tabs ATT: {self.abs_att}" \
                         f"\n\tp. risk: {self.policy_risk}"

        return out_string

    def __lt__(self, other: "SimpleHalvingGridSearchResults") -> bool:

        if self.validation_fold_r2_score < other.validation_fold_r2_score:
            return True
        elif self.validation_fold_r2_score == other.validation_fold_r2_score:

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

            if self.has_counterfactual_score <= 1 and other.has_counterfactual_score <= 1:
                if self.ycf_r2_score < other.ycf_r2_score:
                    return True
                elif self.ycf_r2_score > other.ycf_r2_score:
                    return False

            if self.has_t0_t1_scores and other.has_t0_t1_scores:
                # basically the r2 scores for t1 and t0 are shifted down to
                # have an upper limit of -1, then the products of the
                # shifted r2 scores are found.
                # higher product = worse r2 scores: counted as 'less than'
                # returns true if this object's combined r2 is worse than other.
                if (
                   (self.t0_r2_score-2) * (self.t1_r2_score-2)
                ) > (
                    (other.t0_r2_score-2) * (other.t1_r2_score-2)
                ):
                    return True
                else:
                    return False
            # otherwise compare the factual r2 scores
            return self.yf_r2_score < other.yf_r2_score
        # this is not less than the other thing
        return False

    @property
    def best_estimator_(self) -> PPipeline:
        return self.gridSearch.best_estimator_

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

            importance = self.feature_importances["importances_mean"].values
            error = self.feature_importances["importances_std"].values

            plot_title = f"{self.dataset_name} feature importances using permutation for {self.learner_type}"

            y_label = "Mean accuracy decrease"

        else:

            importance = self.best_estimator_.feature_importances_
            error = self.best_estimator_.feature_importances_std_

            plot_title = f"{self.dataset_name} feature importances using Mean Decrease in Impurity (no permutation) for {self.learner_type}"

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
            figsize=(16,18)
        )

        fig: plt.Figure = faa[0]
        axs: Tuple[plt.Axes, plt.Axes] = faa[1]

        for ax, perm in zip(axs, [False, True]):

            self.plot_feature_importances(ax, perm)

        fig.suptitle(f"{self.dataset_name} feature importance graphs for {self.learner_type}")
        fig.tight_layout()
        fig.savefig(
            fname=os.getcwd() + "\\" + f"{self.dataset_name}" + "\\" + f"{self.dataset_name} {self.learner_type} feature importances.pdf"
        )

        return fig

    def save_me(self) -> NoReturn:

        path: str = os.getcwd() + "\\" + f"{self.dataset_name}" + "\\"

        results_go_here: str = path + f"{self.dataset_name} {self.learner_type} simple results.pickle"

        estimator_goes_here: str = path + f"{self.dataset_name} {self.learner_type} simple estimator.pickle"

        print(f"Pickling results to: {results_go_here}")

        with open(results_go_here, "wb") as resultsPickle:
            pickle.dump(self, resultsPickle)
            print("pickled results!")

        print(f"Pickling simple estimator to {estimator_goes_here}")

        with open(estimator_goes_here, "wb") as estPickle:
            pickle.dump(self.best_estimator_, estPickle)
            print("pickled estimator!")

    @classmethod
    def make_dfm(
            cls,
            searcher: HalvingGridSearchCV,
            df_m: df_utils.DataframeManager,
            learner_type: str
    ) -> "SimpleHalvingGridSearchResults":

        all_x,all_yf = df_m.x_y(
            train=None,
            x_columns=df_utils.DatasetEnum.FACTUAL
        )

        return cls.make(
            searcher=searcher,
            dataset_name=df_m.dataset_name,
            learner_type=learner_type,
            all_x_data=all_x,
            factual_y_data=all_yf,
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
            learner_type: str,
            all_x_data: pd.DataFrame,
            factual_y_data: pd.DataFrame,
            test_indices: T_INDEX,
            e_data: np.ndarray,
            true_t0_t1_ite_ycf: Optional[pd.DataFrame] = None,
            t_column: str = "t",
            yf_column: str = "yf",
            ycf_column:str = "ycf",
            t0_column: str = "t0",
            t1_column: str = "t1",
            ite_column: str = "ite",
    ) -> "SimpleHalvingGridSearchResults":

        this_x: pd.DataFrame = all_x_data.copy()

        this_x_y: pd.DataFrame = this_x.copy()

        train_r2_score: float = searcher.best_score_

        validation_r2_score: float = searcher.score(
            this_x.loc[test_indices],
            factual_y_data.loc[test_indices]
        )

        importance_results: Bunch = permutation_importance(
            searcher,
            this_x.loc[test_indices],
            factual_y_data.loc[test_indices],
            n_repeats=10,
            random_state=seed(),
            n_jobs=-1
        )

        importance_df: pd.DataFrame = pd.DataFrame()

        x_t_names: Tuple[str] = tuple(this_x.columns.values)

        importance_df["features"] = this_x.columns.values

        importance_df["importances_mean"] = importance_results.importances_mean
        importance_df["importances_std"] = importance_results.importances_std
        importance_df["importances"] = importance_results.importances.tolist()

        importance_df.set_index("features")

        this_x_y[yf_column] = searcher.predict(
            this_x.to_numpy()
        )

        factual_r2_score: float = r2_score(
            factual_y_data.to_numpy(),
            this_x_y[yf_column].to_numpy()
        )

        old_t = this_x[t_column].to_numpy(copy=True)

        this_x[t_column] = np.choose(this_x[t_column].values,[1,0])

        this_x_y[ycf_column] = searcher.predict(
            this_x.to_numpy()
        )

        this_x[t_column] = 0

        this_x_y[t0_column] = searcher.predict(this_x.to_numpy())

        this_x[t_column] = 1

        this_x_y[t1_column] = searcher.predict(this_x.to_numpy())

        this_x[t_column] = old_t

        this_x_y[ite_column] = this_x_y[t1_column] - this_x_y[t0_column]

        # dummy values of 42 because  r2 score of 42 is impossible.
        ycf_score: Optional[float] = None
        t0_score: Optional[float] = None
        t1_score: Optional[float] = None

        _abs_ate: Optional[float] = None
        _pehe: Optional[float] = None

        _abs_att: Optional[float] = None
        _sq_att : Optional[float] = None

        _policy_risk: Optional[float] = None

        if true_t0_t1_ite_ycf is not None:
            ycf_score = r2_score(
                true_t0_t1_ite_ycf[ycf_column].to_numpy(),
                this_x_y[ycf_column].to_numpy()
            )
            t0_score = r2_score(
                true_t0_t1_ite_ycf[t0_column].to_numpy(),
                this_x_y[t0_column].to_numpy()
            )
            t1_score = r2_score(
                true_t0_t1_ite_ycf[t1_column].to_numpy(),
                this_x_y[t1_column].to_numpy()
            )

            real_ite = true_t0_t1_ite_ycf[ite_column].to_numpy()
            pred_ite = this_x_y[ite_column].to_numpy()

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
        _t: np.ndarray= this_x_y[t_column].to_numpy()

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
            gridSearch=searcher,
            dataset_name=dataset_name,
            learner_type=learner_type,
            test_indices=test_indices,
            x_t_column_names=x_t_names,
            predictions=this_x_y,
            feature_importances=importance_df,
            gridsearch_train_r2_score=train_r2_score,
            validation_fold_r2_score=validation_r2_score,
            yf_r2_score=factual_r2_score,
            ycf_r2_score=ycf_score,
            t0_r2_score=t0_score,
            t1_r2_score=t1_score,
            abs_ate=_abs_ate,
            pehe=_pehe,
            abs_att=_abs_att,
            policy_risk=_policy_risk,
            t_column=t_column,
            yf_column=yf_column,
            ycf_column=ycf_column,
            t0_column=t0_column,
            t1_column=t1_column,
            ite_column=ite_column
        )



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

    # find the factor with most splits for split_this.
    # for i in range(1, sqrt)

    #floor_sqrt: int = np.floor(np.sqrt(split_this))

    factors: Dict[int, Tuple[int, int]] = dict(
        (i, factor_tester(split_this, i)) for i in range(2, int(np.ceil(np.sqrt(split_this))) + 1)
    )

    best_so_far: Tuple[int, int, int] = (-1, -1, 0)

    for k, v in factors.items():
        if v[1] > best_so_far[2]:
            best_so_far = (k, v[0],v[1])

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

#%%
def simple_nested_halving_grid_searcher(
        estimator: EST,
        param_grid: Dict[str, List[Any]],
        df_m: df_utils.DataframeManager,
        df_selection: df_utils.DatasetEnum,
        learner_name: str,
        kfold_splits: int = 10,
        stratify_on: Iterable[str] = tuple("t"),
        sample_weights: Optional[np.ndarray] = None,
        nested_rng_generator: RNG = None,
        resource: str = "n_samples",
        resource_param_values: Optional[Iterable[int]] = None,
) -> List[SimpleHalvingGridSearchResults]:
    """

    :param estimator:
    :param param_grid:
    :param df_m:
    :param df_selection:
    :param learner_name:
    :param kfold_splits:
    :param stratify_on:
    :param sample_weights:
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


    the_splits: Iterable[Tuple[T_INDEX, T_INDEX]] = df_m.get_kfold_indices(
        train=True,
        random_state=nested_rng_generator,
        class_columns=stratify_on,
        n_splits=kfold_splits
    )

    train_data, train_labels = df_m.x_y(True, x_columns=df_selection)
    #test_data, test_labels = df_m.x_y(False, x_columns=df_selection)

    for i, max_res in enumerate(resource_param_values, 1):

        print(f"-- {i}/{len(resource_param_values)} start --")

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
            max_resources= max_res,
            min_resources= min_res #n_max_resources//4
        )


        if sample_weights is not None:

            current_search.fit(
                train_data, train_labels, sample_weight=sample_weights[df_m.train_indices]
            )
        else:
            current_search.fit(
                train_data, train_labels
            )

        res: SimpleHalvingGridSearchResults = SimpleHalvingGridSearchResults.make_dfm(
            current_search,
            df_m,
            learner_name
        )

        results.append(res)

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


