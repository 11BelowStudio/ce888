# utilities etc for econml

from __future__ import annotations

import econml

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn as sk
import numpy as np

from typing import List, Tuple, Dict, Iterable, Iterator, TypeVar, Union, Optional, NoReturn, Any, Callable


from assignment2.a2_utils.seed_utils import *
from assignment2.a2_utils.metric_utils import *
from assignment2.a2_utils.misc_utils import *
from assignment2.a2_utils import *
import assignment2.a2_utils.dataframe_utils as df_utils
import assignment2.a2_utils.simple_learner_utils as slearner

from sklearn.base import  RegressorMixin, ClassifierMixin, clone

from econml.metalearners import XLearner
from econml.dml import CausalForestDML

import dataclasses

import pickle
import os

@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class CausalForestOutputs:

    cf: CausalForestDML

    dataset_name: str

    learner_name: str

    df_m: df_utils.DataframeManager

    Y: pd.DataFrame
    X: pd.DataFrame
    T: pd.DataFrame

    ytest: pd.DataFrame
    ttest: pd.DataFrame
    xtest: pd.DataFrame

    @classmethod
    def make(
            cls,
            df_m: df_utils.DataframeManager,
            model_t: slearner.PP,
            model_y: slearner.PP,
    ) -> "CausalForestOutputs":

        cf: CausalForestDML = CausalForestDML(
            model_y=clone(model_y),
            model_t=clone(model_t),
            n_estimators=10000,
            min_samples_leaf=10,
            max_depth=None,
            max_samples=0.5,
            discrete_treatment=True,
            honest=True,
            inference=True,
            cv=10,
            random_state=seed(),
            verbose=1
        )

        X, Y = df_m.x_y(train=True, x_columns=df_utils.DatasetEnum.X_Y)

        _, T = df_m.x_y(train=True, x_columns=df_utils.DatasetEnum.X_T)

        xtest, ytest = df_m.x_data(train=False, x_columns=df_utils.DatasetEnum.X_Y)

        _, ttest = df_m.x_y(train=True, x_columns=df_utils.DatasetEnum.X_T)

        cf.fit(Y, T, X=X, W=None)

        return cls(
            cf=cf,
            dataset_name=df_m.dataset_name,
            learner_name="causal forest",
            df_m=df_m,
            Y=Y,
            X=X,
            T=T,
            ttest=ttest,
            ytest=ytest,
            xtest=xtest
        )

    @property
    def train_y_t_x(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.Y, self.T, self.X

    @property
    def test_y_t_x(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.ytest, self.ttest, self.xtest



    def save_me(self) -> NoReturn:

        rel_folder: str = f"\\{self.dataset_name}\\"

        results_go_here_rel: str = f"{rel_folder}{self.dataset_name} {self.learner_name} results.pickle"

        print(f"Pickling self to: {results_go_here_rel}")

        with open(f"{os.getcwd()}{results_go_here_rel}", "wb") as resultsPickle:
            pickle.dump(self, resultsPickle)
            print("pickled results!")


    @classmethod
    def load_me(cls, save_location: str) -> "CausalForestOutputs":

        with open(save_location, "rb") as load_pickle:
            return pickle.load(load_pickle, fix_imports=True)



@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class EconLearnerOutputs:
    "recycled from my lab 4 work"

    learner: Union[slearner.R, XLearner]
    ite_pred: np.ndarray
    abs_ate: float
    pehe: float

    @classmethod
    def make(
            cls,
            regressor: slearner.R,
            xt_training,
            y_training,
            xt0_testing,
            xt1_testing,
            ite_real
    ):

        reg = regressor.fit(xt_training, y_training)

        t0 = reg.predict(xt0_testing)
        t1 = reg.predict(xt1_testing)
        ite = t1 - t0

        return cls(reg, ite, abs_ate(ite_real, ite), pehe(ite_real, ite))

    @classmethod
    def make_sampleweights(cls, regressor, xt_training, y_training, xt0_testing, xt1_testing, weights, ite_real):

        reg = regressor.fit(xt_training, y_training, sample_weight=weights)
        t0 = reg.predict(xt0_testing)
        t1 = reg.predict(xt1_testing)
        ite = t1 - t0
        return cls(reg, ite, abs_ate(ite_real, ite), pehe(ite_real, ite))

    @classmethod
    def make_xlearner(cls, xl_models, prop_model, x_training, y_training, t_training, x_testing, ite_real, cate_models=None):
        xl: XLearner = XLearner(models=xl_models, propensity_model=prop_model, cate_models=cate_models)
        xl.fit(y_training, t_training.flatten(), X=x_training)
        ite = xl.effect(x_testing)
        return cls(xl,ite, abs_ate(ite_real, ite), pehe(ite_real, ite))

    @classmethod
    def make_cate(cls, cate_model, x_training, y_training, t_training, x_testing, ite_real):
        cate_model.fit(y_training, t_training.flatten(), X=x_training)
        ite = cate_model.effect(x_testing)
        return cls(cate_model, ite, abs_ate(ite_real, ite), pehe(ite_real, ite))

    def ate(self, effect_true):
        return abs_ate(effect_true, self.ite_pred)

    def calc_pehe(self, effect_true):
        return pehe(effect_true, self.ite_pred)

