# utilities etc for econml

from __future__ import annotations



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Iterable, Iterator, TypeVar, Union, Optional, NoReturn, Any, Callable, ClassVar, \
    Final

import sklearn.base

from assignment2.a2_utils.seed_utils import *
from assignment2.a2_utils.metric_utils import *
from assignment2.a2_utils.misc_utils import *
from assignment2.a2_utils import *
import assignment2.a2_utils.dataframe_utils as df_utils
import assignment2.a2_utils.simple_learner_utils as slearn

from assignment2.a2_utils.simple_learner_utils import PP, PPipeline

from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegressionCV

import shap
from shap._explanation import Explanation

import sys
import math

import econml
from econml._cate_estimator import BaseCateEstimator, LinearCateEstimator
from econml._ortho_learner import _OrthoLearner as OrthoLearner
from econml.metalearners import XLearner
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner
from econml.iv.dr import ForestDRIV
from econml.iv.dml import DMLIV
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter

from functools import cached_property
from sklearn.model_selection import KFold, StratifiedKFold

from enum import Enum, auto

import copy

import dataclasses

import pickle
import os

npdf = Union[pd.DataFrame, np.ndarray]
"alias for something that can be a dataframe or an ndarray"

fi = Union[int, float]
"alias for int or float"


@dataclasses.dataclass(init=True, repr=True, frozen=True)
class XYT:
    "convenience class for returning X, Y, and T data."

    X: npdf
    Y: npdf
    T: npdf
    E: npdf

    @property
    def kw(self) -> Dict[str, npdf]:
        "returns this as a dict of X, Y, T, which can be unpacked (with **) and used as kwargs X,Y,T"
        return {
            "X": self.X,
            "Y": self.Y,
            "T": self.T
        }

    def kw_z(self, instrumented: bool = False, e_kw: str = "Z") -> Dict[str, npdf]:
        """
        returns this as a dict of X,Y,T and (potentially) e_kw, if instrumented=True
        :param instrumented: If false, returns dict of XYT. If true, returns dict of X,Y,Z,e_kw
        :param e_kw: keyword to assign to E. Defaults to 'Z'
        :return: either a dict of XYT (default) or a dict of X,Y,T,e_kw if instrumented=True
        """
        if instrumented:
            return self.kw_e(e_kw)
        else:
            return self.kw

    def kw_e(self, e_kw: str = "E") -> Dict[str, npdf]:
        """
        Returns this as a dict of X, Y, T, e_kw, unpackable with **, with and usable as kwargs X,Y,T,e_kw
        :param e_kw: the kwarg keyword for self.E
        :return: this as a dict, intended to be unpacked for use with kwargs
        """
        return {
            "X": self.X,
            "Y": self.Y,
            "T": self.T,
            e_kw: self.E
        }

    @classmethod
    def make(cls, X: npdf, Y: npdf, T: npdf, E: Optional[npdf] = None) -> "XYT":
        return cls(
            X,
            Y,
            T,
            np.ones_like(T) if E is None else E
        )


class InstrumentIVEnum(Enum):

    T_XW = auto()
    "for the T|X,W classifier for IV"
    T_XWZ = auto()
    "for the T|X,W,Z classifier"
    TZ_XW = auto()
    "for the T*Z|X,W classifier"
    Z_XW = auto()
    "for the Z|X,W classifier"

    @classmethod
    def param_names(cls) -> Iterable[Tuple[str, "InstrumentIVEnum"]]:
        return zip(
            ("model_t_xw","model_z_xw","model_t_xwz","model_tz_xw"),
            (cls.T_XW, cls.Z_XW, cls.T_XWZ, cls.TZ_XW)
        )



@dataclasses.dataclass(init=True, repr=True, frozen=True)
class InstrumentedValueModel(ClassifierMixin):

    model_type: InstrumentIVEnum
    inner_model: PP
    inner_z: Optional[PP]

    @classmethod
    def IV_params(cls, ipsw: slearn.IpswWrapper) -> Dict[str, "InstrumentedValueModel"]:
        return dict(
            (pname, cls.make(ipsw, pval)) for pname, pval in InstrumentIVEnum.param_names()
        )

    @classmethod
    def make(cls, ipsw: slearn.IpswWrapper, mtype: InstrumentIVEnum) -> "InstrumentedValueModel":

        if (mtype == InstrumentIVEnum.TZ_XW) or (mtype == InstrumentIVEnum.Z_XW):
            # noinspection PyTypeChecker
            return cls(
                model_type=mtype,
                inner_model=sklearn.base.clone(ipsw.ipsw_clf),
                inner_z=PPipeline.make(estimator=LogisticRegressionCV(
                    cv=3,
                    penalty="elasticnet",
                    l1_ratios=[0,0.5,1],
                    n_jobs=-1,
                    solver="saga",
                    class_weight="balanced"
                ))
            )

        # noinspection PyTypeChecker
        return cls(
            model_type=mtype,
            inner_model=sklearn.base.clone(ipsw.ipsw_clf),
            inner_z=None
        )

    def _validate_data(self, **kwargs):
        return super()._v

    def fit(
        self, X: npdf, W: Optional[npdf]=None, y: Optional[npdf]=None, *, Z: Optional[npdf]=None, Y:Optional[npdf]=None, **kwargs
    ) -> PP:



        if Y is None:
            if y is None:
                if W is not None:
                    Y = W
                    W = None
            else:
                Y = y



        assert Y is not None

        xz_splits: List[np.ndarray] = []

        if isinstance(X, pd.DataFrame):
            xz_splits.extend(np.hsplit(ary=X.to_numpy(), indices_or_sections=X.shape[1]))
        else:
            xz_splits.extend(np.hsplit(ary=X, indices_or_sections=X.shape[1]))

        if W is None:
            W: npdf = np.ones_like(Y.shape[0], dtype=int)

        if self.model_type == InstrumentIVEnum.T_XW:
            self.inner_model.fit(X=X, y=Y)
        elif self.model_type == InstrumentIVEnum.T_XWZ:
            #assert False
            self.inner_model.fit(X=np.hstack(xz_splits[:-1]), y=Y)
        elif self.model_type == InstrumentIVEnum.TZ_XW:
            #assert False
            self.inner_model.fit(X=np.hstack(xz_splits[:-1]), y=Y)
            self.inner_z.fit(X=np.hstack(xz_splits[:-1]), y=xz_splits[-1])
        elif self.model_type == InstrumentIVEnum.Z_XW:
            self.inner_z.fit(X=X, y=Y)

        return self

    def predict(
        self, XWZ: npdf, **kwargs
    ) -> npdf:

        xz_splits: List[np.ndarray] = []

        if isinstance(XWZ, pd.DataFrame):
            xz_splits.extend(np.hsplit(ary=XWZ.to_numpy(), indices_or_sections=XWZ.shape[1]))
        else:
            xz_splits.extend(np.hsplit(ary=XWZ, indices_or_sections=XWZ.shape[1]))

        if self.model_type == InstrumentIVEnum.T_XW:
            return self.inner_model.predict(XWZ).astype(dtype=bool)
        elif self.model_type == InstrumentIVEnum.T_XWZ:
            return (self.inner_model.predict(np.hstack(xz_splits[:-1])) * xz_splits[-1]).astype(dtype=bool)
        elif self.model_type == InstrumentIVEnum.Z_XW:
            return self.inner_z.predict(XWZ).astype(dtype=bool)
        elif self.model_type == InstrumentIVEnum.TZ_XW:
            return (self.inner_model.predict(np.hstack(xz_splits[:-1]))  * self.inner_z.predict(xz_splits[-1])).astype(dtype=bool)

        return np.zeros(XWZ.shape[0]).astype(dtype=bool)

    def predict_proba(
            self, XWZ: npdf, **kwargs
    ) -> npdf:

        xz_splits: List[np.ndarray] = []

        if isinstance(XWZ, pd.DataFrame):
            xz_splits.extend(np.hsplit(ary=XWZ.to_numpy(), indices_or_sections=XWZ.shape[1]))
        else:
            xz_splits.extend(np.hsplit(ary=XWZ, indices_or_sections=XWZ.shape[1]))

        if self.model_type == InstrumentIVEnum.T_XW:
            return self.inner_model.predict_proba(XWZ)
        elif self.model_type == InstrumentIVEnum.T_XWZ:
            return self.inner_model.predict_proba(np.hstack(xz_splits[:-1])) * xz_splits[-1]
        elif self.model_type == InstrumentIVEnum.Z_XW:
            return self.inner_z.predict_proba(XWZ)
        elif self.model_type == InstrumentIVEnum.TZ_XW:
            return self.inner_model.predict_proba(np.hstack(xz_splits[:-1])) * self.inner_z.predict_proba(xz_splits[-1])

        return np.zeros(XWZ.shape[0])






@dataclasses.dataclass(init=True, repr=True, frozen=True)
class EconDFM:
    "A convenient wrapper for the DataframeManager."

    dfm: df_utils.DataframeManager
    _ipsw: slearn.IpswWrapper

    has_counterfactuals: bool

    @classmethod
    def make(cls, dfm: df_utils.DataframeManager, ipsw: slearn.IpswWrapper) -> "EconDFM":

        return cls(
            dfm,
            _ipsw=ipsw,
            has_counterfactuals=dfm.ycf_column is not None
        )

    def getfrom(
            self, train: Optional[Union[bool, pd.DataFrame]], cols: df_utils.DatasetEnum, ind: int
    ) -> pd.DataFrame:
        return self.dfm.x_y(train=train, x_columns=cols)[ind]

    @cached_property
    def X(self) -> pd.DataFrame:
        return self.getfrom(True, df_utils.DatasetEnum.X_Y, 0)

    @cached_property
    def Y(self) -> pd.DataFrame:
        return self.getfrom(True, df_utils.DatasetEnum.X_Y, 1)

    @cached_property
    def T(self) -> pd.DataFrame:
        return self.getfrom(True, df_utils.DatasetEnum.X_T, 1)

    @cached_property
    def xtest(self) -> pd.DataFrame:
        return self.getfrom(False, df_utils.DatasetEnum.X_Y, 0)

    @property
    def ytest(self) -> pd.DataFrame:
        return self.getfrom(False, df_utils.DatasetEnum.X_Y, 1)

    @property
    def ttest(self) -> pd.DataFrame:
        return self.getfrom(False, df_utils.DatasetEnum.X_T, 1)

    @property
    def xall(self) -> pd.DataFrame:
        return self.getfrom(None, df_utils.DatasetEnum.X_Y, 0)

    @property
    def yall(self) -> pd.DataFrame:
        return self.getfrom(None, df_utils.DatasetEnum.X_Y, 1)

    @property
    def tall(self) -> pd.DataFrame:
        return self.getfrom(None, df_utils.DatasetEnum.X_T, 1)

    @property
    def tcf(self) -> pd.DataFrame:
        return self.getfrom(None, df_utils.DatasetEnum.IPSW_X_TCF, 1)

    @property
    def ycf(self) -> Optional[pd.DataFrame]:
        if self.has_counterfactuals:
            return self.getfrom(None, df_utils.DatasetEnum.COUNTERFACTUAL, 1)
        print(f"Dataset {self.dfm.dataset_name} has no counterfactuals!", file=sys.stderr)

    def _append_xyt_counterfactuals(
            self, x: pd.DataFrame, y: pd.DataFrame, t: pd.DataFrame, e: Optional[pd.DataFrame] = None
    ) -> XYT:

        if self.has_counterfactuals:
            x = pd.concat([x, self.xall], ignore_index=True)
            y = pd.concat([y, self.ycf], ignore_index=True)
            t = pd.concat([t, self.tcf], ignore_index=True)
            if e is not None:
                e = pd.concat([e, self.dfm.get_e(None)], ignore_index=True)

        return XYT.make(X=x, Y=y, T=t, E=e)

    @property
    def train_xyt(self) -> XYT:
        return XYT.make(
            X=self.X,
            Y=self.Y,
            T=self.T,
            E=self.dfm.get_e(True)
        )

    @property
    def test_xyt(self) -> XYT:
        return XYT.make(
            X=self.xtest,
            Y=self.ytest,
            T=self.ttest,
            E=self.dfm.get_e(False)
        )

    @property
    def xyt(self) -> XYT:
        return XYT.make(
            X=self.xall,
            Y=self.yall,
            T=self.tall,
            E=self.dfm.get_e(None)
        )

    @property
    def all_test_xyt(self) -> XYT:
        """:return: the test XYT along with the counterfactual XYT (if counterfactuals exist)"""
        return self._append_xyt_counterfactuals(
            x=self.xtest,
            y=self.ytest,
            t=self.ttest,
            e=self.dfm.get_e(False)
        )

    @property
    def all_xyt(self) -> XYT:
        """:return: all the XYT data (all factuals and counterfactuals)"""
        return self._append_xyt_counterfactuals(
            x=self.xall,
            y=self.yall,
            t=self.tall,
            e=self.dfm.get_e(None)
        )

    @property
    def iv_params(self) -> Dict[str, InstrumentedValueModel]:
        return InstrumentedValueModel.IV_params(self._ipsw)

    def ipsw(self, x, t) -> np.ndarray:
        return self._ipsw.ipsw(x, t)

    @property
    def ipsw_wrapper(self) -> slearn.IpswWrapper:
        return self._ipsw

    @property
    def propensity_model(self) -> slearn.PP:
        """:returns: a clone of the propensity model"""
        # noinspection PyTypeChecker
        return sklearn.base.clone(self._ipsw.ipsw_clf)


class RegressorWrapper(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def __init__(self, *, estimator: PP):
        super().__init__()
        self._est: PP = estimator


    def fit(self, X, y, **kwargs):
        return self._est.fit(X,y,**kwargs)

    def predict(self, X, **predict_params):
        return self._est.predict(X, **predict_params)

    def set_params(self, **params):
        return self._est.set_params(**params)

    def get_params(self, deep=True):
        return self._est.get_params(deep),

    def __repr__(self, N_CHAR_MAX=700) -> str:
        return self._est.__repr__(N_CHAR_MAX)

    def __getstate__(self):
        return self._est.__getstate__()

    def __setstate__(self, state) -> NoReturn:
        self._est.__setstate__(state)

    def _more_tags(self):
        return self._est._more_tags()

    def _get_tags(self):
        return self._est._get_tags()

    def _check_n_features(self, X, reset):
        return self._est._check_n_features(X, reset)

    def _check_feature_names(self, X, *, reset):
        return self._est._check_feature_names(X=X, reset=reset)

    def _validate_data(
            self,
            X="no_validation",
            y="no_validation",
            reset=True,
            validate_separately=False,
            **check_params,
    ):
        return self._est._validate_data(X,y,reset,validate_separately, **check_params)

    @property
    def _repr_html_(self):
        return self._est._repr_html_()

    def _repr_html_inner(self):
        return self._est._repr_html_inner()

    def _repr_mimebundle_(self, **kwargs):
        return self._est._repr_mimebundle_(**kwargs)


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class EconMLOutputs:

    cate_est: LinearCateEstimator
    dataset_name: str
    learner_name: str
    _edfm: ClassVar[EconDFM]

    instrumented: bool

    effect: np.ndarray

    @classmethod
    def _make(
            cls,
            cate_est: LinearCateEstimator,
            edfm: EconDFM,
            learner_name: str,
            instrumented: bool = False,
            inference: Optional[str] = None
    ) -> "EconMLOutputs":

        try:
            if cls._edfm is None:
                cls._edfm = edfm
        except AttributeError as e:
            cls._edfm = edfm

        # noinspection PyBroadException
        try:
            if inference is not None:
                cate_est.fit(
                    **edfm.train_xyt.kw_z(instrumented),
                    cache_values=True,
                    inference=inference
                )
            else:
                cate_est.fit(
                    **edfm.train_xyt.kw_z(instrumented),
                    cache_values=True
                )
        except Exception as e1:
            assert edfm.X is not None
            cate_est.fit(
                X=edfm.X,
                Y=edfm.Y,
                T=edfm.T
            )

        return cls(
            cate_est=cate_est,
            learner_name=learner_name,
            dataset_name=edfm.dfm.dataset_name,
            effect=cate_est.effect(
                X=edfm.xall,
                T0=np.zeros_like(edfm.tall).flatten(),
                T1=np.ones_like(edfm.tall).flatten()
            ),
            instrumented=instrumented
        )

    @classmethod
    def make_causal_forest(
            cls,
            edfm: EconDFM,
            model_y: slearn.PP,
            model_t: slearn.PP,
            criterion: Literal["het","mse"]= 'het',
            n_estimators: int = 1000,
            min_samples_leaf: fi = 5,
            max_depth: Optional[int] = None,
            min_samples_split: fi = 10,
            max_samples: fi = 0.5,
            discrete_treatment: bool=True,
            honest: bool =True,
            inference: bool =True,
            cv: Optional[Union[int, KFold, StratifiedKFold, Iterable[Tuple[df_utils.T_INDEX,df_utils.T_INDEX]]]] = 5,
            random_state: RNG = seed(),
            verbose: int = 0,
            subforest_size: int = 4,
            n_jobs: int = -1,
            mc_iters: Optional[int] = None,
            mc_agg: Optional[Literal["mean","median"]] = 'mean',
            drate: bool = True,
            min_weight_fraction_leaf: Optional[float] = 0.,
            min_var_fraction_leaf: Optional[float] = None,
            min_var_leaf_on_val: Any = False,
            max_features: Any = "auto",
            min_impurity_decrease: Any = 0.,
            min_balancedness_tol: Any = .45,
            fit_intercept: Any = True,
            **kwargs
    ) -> "EconMLOutputs":

        return cls._make(
            cate_est=CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                criterion=criterion,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_samples=max_samples,
                discrete_treatment=discrete_treatment,
                honest=honest,
                inference=inference,
                cv=cv,
                random_state=random_state,
                verbose=verbose,
                subforest_size=subforest_size,
                n_jobs=n_jobs,
                mc_iters=mc_iters,
                mc_agg=mc_agg,
                drate=drate,
                min_samples_split=min_samples_split,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                min_var_fraction_leaf=min_var_fraction_leaf,
                min_var_leaf_on_val=min_var_leaf_on_val,
                max_features=max_features,
                min_impurity_decrease=min_impurity_decrease,
                min_balancedness_tol=min_balancedness_tol,
                fit_intercept=fit_intercept
            ),
            edfm=edfm,
            learner_name="Causal Forest",
            instrumented=False,
            inference="auto"
        )

    @classmethod
    def make_ForestDR(
            cls,
            edfm: EconDFM,
            model_regression: slearn.PP,
            model_propensity: slearn.PP,
            featurizer: Optional[sklearn.base.TransformerMixin]=None,
            min_propensity: Optional[float]=1e-6,
            categories='auto',
            cv: Optional[Union[int, KFold, StratifiedKFold, Iterable[Tuple[df_utils.T_INDEX,df_utils.T_INDEX]]]] = 5,
            mc_iters: Optional[int] =None,
            mc_agg: Optional[Literal["mean","median"]]='mean',
            n_estimators: int=1000,
            max_depth: Optional[int] =None,
            min_samples_split: Optional[fi]= 10,
            min_samples_leaf: Optional[fi]= 5,
            min_weight_fraction_leaf: Optional[float]=0.,
            max_features: Optional[Union[fi, str]]= "auto",
            min_impurity_decrease: Optional[float]= 0.,
            max_samples: fi = .45,
            min_balancedness_tol: float=.45,
            honest: bool = True,
            subforest_size: int = 4,
            n_jobs: int = -1,
            verbose: int = 0,
            random_state: RNG = seed(),
            **kwargs
    ) -> "EconMLOutputs":

        return cls._make(
            cate_est=ForestDRLearner(
                model_regression=model_regression,
                model_propensity=model_propensity,
                featurizer=featurizer,
                min_propensity=min_propensity,
                categories=categories,
                cv=cv,
                mc_iters=mc_iters,
                mc_agg=mc_agg,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                min_impurity_decrease=min_impurity_decrease,
                max_samples=max_samples,
                min_balancedness_tol=min_balancedness_tol,
                honest=honest,
                subforest_size=subforest_size,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state
            ),
            edfm=edfm,
            learner_name="Forest DR Learner",
            instrumented=False,
            inference="bootstrap"
        )

    @classmethod
    def make_DMLIV(
            cls,
            edfm: EconDFM,
            model_y_xw: slearn.PP,
            model_final: Optional[slearn.R] = None,
            featurizer: Optional[sklearn.base.TransformerMixin]=None,
            fit_cate_intercept: bool=True,
            discrete_treatment: bool=True,
            discrete_instrument: bool=True,
            categories='auto',
            cv: Optional[Union[int, KFold, StratifiedKFold, Iterable[Tuple[df_utils.T_INDEX,df_utils.T_INDEX]]]] = 5,
            mc_iters: Optional[int] = None,
            mc_agg: Optional[Literal["mean","median"]]='mean',
            random_state: RNG = seed(),
            **kwargs
    ) -> "EconMLOutputs":
        return cls._make(

            cate_est=DMLIV(
                model_y_xw=model_y_xw,
                model_t_xw=edfm.propensity_model,  #InstrumentedValueModel.make(edfm.ipsw_wrapper,InstrumentIVEnum.T_XW),
                model_t_xwz=PPipeline.make_simpler(estimator=LogisticRegressionCV(
                    cv=3,
                    penalty="elasticnet",
                    l1_ratios=[0,0.5,1],
                    n_jobs=-1,
                    solver="saga",
                    class_weight="balanced"
                )),                 #InstrumentedValueModel.make(edfm.ipsw_wrapper,InstrumentIVEnum.T_XWZ),
                model_final=model_final if model_final is not None else econml.sklearn_extensions.linear_model.StatsModelsLinearRegression(fit_intercept=False),
                featurizer=featurizer,
                fit_cate_intercept=fit_cate_intercept,
                discrete_treatment=discrete_treatment,
                discrete_instrument=discrete_instrument,
                categories=categories,
                cv=cv,
                mc_iters=mc_iters,
                mc_agg=mc_agg,
                random_state=random_state
            ),
            edfm=edfm,
            learner_name="DMLIV",
            instrumented=True,
            inference="bootstrap"
        )

    @classmethod
    def make_ForestDRIV(
            cls,
            edfm: EconDFM,
            model_y_xw: slearn.PP,
            model_z_xw: Optional[Union[slearn.PP, slearn.R]] = None,
            flexible_model_effect="auto",
            prel_cate_approach="driv",
            prel_cv=1,
            prel_opt_reweighted=True,
            projection: bool =False,
            featurizer=None,
            n_estimators=1000,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            min_impurity_decrease=0.,
            max_samples=.45,
            min_balancedness_tol=.25,
            honest=True,
            subforest_size=4,
            n_jobs=-1,
            verbose=0,
            cov_clip=1e-3,
            opt_reweighted=False,
            discrete_instrument=False,
            discrete_treatment=False,
            categories='auto',
            cv=5,
            mc_iters=None,
            mc_agg='mean',
            random_state: RNG = seed()
    )-> "EconMLOutputs":
        return cls._make(
            cate_est=ForestDRIV(
                model_y_xw=model_y_xw,
                model_t_xw=edfm.propensity_model,
                model_t_xwz="auto" if (projection==False and prel_cate_approach == "driv") else PPipeline.make_simpler(
                    estimator=LogisticRegressionCV(
                        cv=5,
                        penalty="elasticnet",
                        l1_ratios=[0,0.5,1],
                        n_jobs=1,
                        solver="saga",
                        class_weight="balanced"
                    )
                ),
                model_z_xw=model_z_xw if model_z_xw is not None else PPipeline.make_simpler(estimator=LogisticRegressionCV(
                    cv=5,
                    penalty="elasticnet",
                    l1_ratios=[0,0.5,1],
                    n_jobs=1,
                    solver="saga",
                    class_weight="balanced"
                )),
                model_tz_xw=PPipeline.make_simpler(estimator=LogisticRegressionCV(
                    cv=5,
                    penalty="elasticnet",
                    l1_ratios=[0,0.5,1],
                    n_jobs=1,
                    solver="saga",
                    class_weight="balanced"
                )),
                flexible_model_effect=flexible_model_effect,
                prel_cate_approach=prel_cate_approach,
                prel_cv=prel_cv,
                prel_opt_reweighted=prel_opt_reweighted,
                projection=projection,
                featurizer=featurizer,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                min_impurity_decrease=min_impurity_decrease,
                max_samples=max_samples,
                min_balancedness_tol=min_balancedness_tol,
                honest=honest,
                subforest_size=subforest_size,
                n_jobs=n_jobs,
                verbose=verbose,
                cov_clip=cov_clip,
                opt_reweighted=opt_reweighted,
                discrete_instrument=discrete_instrument,
                discrete_treatment=discrete_treatment,
                categories=categories,
                cv=cv,
                mc_iters=mc_iters,
                mc_agg=mc_agg,
                random_state=random_state
            ),
            edfm=edfm,
            learner_name="ForestDRIV",
            instrumented=True,
            inference="bootstrap"
        )

    @classmethod
    def make_XLearner(
            cls,
            edfm: EconDFM,
            xt_y_models: Union[slearn.PP, List[slearn.PP]],
            cate_models: Optional[Union[BaseCateEstimator, List[BaseCateEstimator]]] = None,
            categories="auto"
    ) -> "EconMLOutputs":

        if isinstance(xt_y_models, sklearn.base.BaseEstimator):
            # noinspection PyTypeChecker
            xt_y_models: slearn.PP = sklearn.base.clone(xt_y_models)
        else:
            # noinspection PyTypeChecker
            xt_y_models: List[slearn.PP] = [sklearn.base.clone(e) for e in xt_y_models]

        if cate_models is not None:
            if isinstance(cate_models, BaseCateEstimator):
                cate_models: BaseCateEstimator = copy.deepcopy(cate_models)
            else:
                cate_models: List[BaseCateEstimator] = [copy.deepcopy(c) for c in cate_models]

        return cls._make(
            cate_est=XLearner(
                models=xt_y_models,
                cate_models=cate_models,
                propensity_model=edfm.propensity_model,
                categories=categories
            ),
            edfm=edfm,
            learner_name="XLearner",
            instrumented=False
        )

    @property
    def edfm(self) -> EconDFM:
        return self.__class__._edfm

    @property
    def est(self) -> BaseCateEstimator:
        return self.cate_est

    @property
    def get_est(self) -> BaseCateEstimator:
        return copy.deepcopy(self.est)

    @cached_property
    def mean_squared_error(self) -> Optional[float]:
        try:
            # noinspection PyTypeChecker
            ol: OrthoLearner = self.cate_est
            return ol.score(
                **self.edfm.all_test_xyt.kw_z(instrumented=self.instrumented)
            )
        except Exception as e:
            return None

    @property
    def has_mean_squared_error(self) -> bool:
        return self.mean_squared_error is not None

    def __lt__(self, other: "EconMLOutputs") -> bool:

        if other is None:
            return False

        if self.edfm.has_counterfactuals and other.edfm.has_counterfactuals:
            # start off by comparing PEHE (lower=better, so higher pehe = less than other)
            if self.pehe != other.pehe:
                return self.pehe > other.pehe

            elif self.pehe_all != other.pehe_all:
                return self.pehe_all > other.pehe_all

        if self.abs_att != other.abs_att:
            return self.abs_att > other.abs_att

        if self.has_mean_squared_error and other.has_mean_squared_error:
            if self.mean_squared_error != other.mean_squared_error:
                return self.mean_squared_error > other.mean_squared_error

        if self.policy_risk != other.policy_risk:
            return self.policy_risk > other.policy_risk

        elif self.policy_risk_all != other.policy_risk_all:
            return self.policy_risk_all > other.policy_risk_all

        else:
            return self.abs_att > other.abs_att

    @property
    def info(self) -> str:
        """:return: returns a string with some info about this object"""
        out_str: str = f"Info for {self.learner_name} CATE estimator on {self.dataset_name}:"

        if self.has_mean_squared_error:
            out_str = f"{out_str}" \
                      f"\n\tError:  {self.mean_squared_error} [mean squared error for XYTZ]"
        if self.edfm.has_counterfactuals:
            # noinspection PyPep8 E222
            # noinspection E222
            out_str = f"{out_str}" \
                      f"\n\tPEHE:   {self.pehe}\t[Precision in Estimation of Heterogeneous treatment Effect]" \
                      f"\n\tATE:    {self.abs_ate}\t[Absolute error for Average Treatment Effect]"  # noqa: E222
        out_str = f"{out_str}" \
                  f"\n\tATT:    {self.abs_att}\t[Absolute error for Average Treatment effect on the Treated]" \
                  f"\n\tP. Risk:{self.policy_risk}\t[Policy Risk]"  # noqa: E222

        return out_str

    @cached_property
    def _shap_values(self) -> Dict[str, Dict[str, Explanation]]:
        """:return: shap values for this estimator"""
        return self.cate_est.shap_values(self.edfm.xall, feature_names=self.edfm.xall.columns.values)

    @property
    def shap_values(self) -> Explanation:
        """:return: the shap values for this estimator but not nested"""
        sv = self._shap_values
        outer_v: Dict[str, Explanation] = [*sv.values()][0]
        return [*outer_v.values()][0]

    def shap_plot(self) -> plt.Figure:
        """
        Creates a figure and plots the SHAP values (feature importances) on it
        :return: the figure with the SHAP values plotted on it.
        """
        shap.summary_plot(
            self.shap_values,
            plot_size=(16, 16),
            cmap="coolwarm",
            show=False,
            max_display=len(self.edfm.xall.columns.tolist())
        )
        fig: plt.Figure = plt.gcf()
        fig.suptitle(f"SHAP values (effect on Y) for each feature in {self.dataset_name}")
        fig.set_tight_layout(tight=True)

        save_here: str = f"{self.dataset_name}\\{self.dataset_name} SHAP values for {self.learner_name}.pdf"

        print(f"Saving SHAP figure to {save_here}...")
        fig.savefig(fname=f"{os.getcwd()}\\{save_here}")

        return fig

    @cached_property
    def policy_risk_all(self) -> float:
        return metric_utils.policy_risk(
            effect_pred=self.cate_est.effect(self.edfm.xall),
            yf=self.edfm.yall.values,
            t=self.edfm.tall.values,
            e=self.edfm.dfm.get_e(train=None).to_numpy().flatten()
        )

    @cached_property
    def policy_risk(self) -> float:
        return metric_utils.policy_risk(
            effect_pred=self.cate_est.effect(self.edfm.xtest),
            yf=self.edfm.ytest.values,
            t=self.edfm.ttest.values,
            e=self.edfm.dfm.get_e(train=False).to_numpy().flatten()
        )

    @cached_property
    def abs_att(self) -> float:
        return metric_utils.abs_att(
            effect_pred=self.cate_est.effect(self.edfm.xtest),
            yf=self.edfm.ytest.values,
            t=self.edfm.ttest.values,
            e=self.edfm.dfm.get_e(train=False).to_numpy().flatten()
        )

    @cached_property
    def abs_att_all(self) -> float:
        return metric_utils.abs_att(
            effect_pred=self.cate_est.effect(self.edfm.xall),
            yf=self.edfm.yall.values,
            t=self.edfm.tall.values,
            e=self.edfm.dfm.get_e(train=None).to_numpy().flatten()
        )

    @cached_property
    def abs_ate(self) -> float:
        if self.edfm.has_counterfactuals:
            return metric_utils.abs_ate(
                effect_true=self.edfm.dfm.x_data(train=False, x_columns=[self.edfm.dfm.ite_column]).values,
                effect_pred=self.cate_est.effect(
                    X=self.edfm.xtest,
                    T0=np.zeros_like(self.edfm.ttest),
                    T1=np.ones_like(self.edfm.ttest)
                )
            )
        else:
            print(f"Dataset {self.dataset_name} has no counterfactuals, cannot measure ATE (average treatment effect)!",
                  file=sys.stderr)
            return math.inf

    @cached_property
    def abs_ate_all(self) -> float:
        if self.edfm.has_counterfactuals:
            return metric_utils.abs_ate(
                effect_true=self.edfm.dfm.x_data(train=None, x_columns=[self.edfm.dfm.ite_column]).values,
                effect_pred=self.cate_est.effect(
                    X=self.edfm.xall,
                    T0=np.zeros_like(self.edfm.tall),
                    T1=np.ones_like(self.edfm.tall)
                )
            )
        else:
            print(
                f"Dataset {self.dataset_name} has no counterfactuals, cannot measure ATE (average treatment effect)!",
                file=sys.stderr)
            return math.inf

    @cached_property
    def pehe(self) -> float:
        if self.edfm.has_counterfactuals:
            return metric_utils.pehe(
                effect_true=self.edfm.dfm.x_data(train=False, x_columns=[self.edfm.dfm.ite_column]).values,
                effect_pred=self.cate_est.effect(self.edfm.xtest)
            )
        else:
            print(f"Dataset {self.dataset_name} has no counterfactuals, cannot measure PEHE "
                  f"(precision of heterogeneous treatment effect)!", file=sys.stderr)
            return math.inf

    @cached_property
    def pehe_all(self) -> float:
        if self.edfm.has_counterfactuals:
            return metric_utils.pehe(
                effect_true=self.edfm.dfm.x_data(train=None, x_columns=[self.edfm.dfm.ite_column]).values,
                effect_pred=self.cate_est.effect(self.edfm.xall)
            )
        else:
            print(f"Dataset {self.dataset_name} has no counterfactuals, cannot measure PEHE!", file=sys.stderr)
            return math.inf

    def cate_tree(
            self,
            train: Optional[bool] = False,
            include_model_uncertainty: bool = True,
            uncertainty_level=0.05,
            uncertainty_only_on_leaves=True,
            splitter: str = "best",
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features: Union[str, int, float] = "auto",
            random_state=seed(),
            max_leaf_nodes: Optional[int] = None,
            min_impurity_decrease: float = 0.0
    ) -> SingleTreeCateInterpreter:
        """
        Attempts to create a SingleTreeCateInterpreter.

        For all params except 'train',
        see https://econml.azurewebsites.net/_autosummary/econml.cate_interpreter.SingleTreeCateInterpreter.html
        :param train: false to interpret test set, true to interpret training set, None to interpret full data
        :param include_model_uncertainty:
        :param uncertainty_level:
        :param uncertainty_only_on_leaves:
        :param splitter:
        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param min_weight_fraction_leaf:
        :param max_features:
        :param random_state:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
        :return:
        """
        ctree: SingleTreeCateInterpreter = SingleTreeCateInterpreter(
            include_model_uncertainty=include_model_uncertainty,
            uncertainty_level=uncertainty_level,
            uncertainty_only_on_leaves=uncertainty_only_on_leaves,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease
        )

        return ctree.interpret(
            self.cate_est,
            self.edfm.dfm.x_data(train=train, x_columns=df_utils.DatasetEnum.X_Y)
        )

    def policy_tree(
            self,
            train: Optional[bool] = False,
            include_model_uncertainty=True,
            uncertainty_level=0.05,
            uncertainty_only_on_leaves=True,
            risk_level: Optional[float] = 0.05,
            risk_seeking: bool = True,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="auto",
            random_state=seed(),
            min_impurity_decrease=0.0,
            min_balancedness_tol: float = 0.25,
            sample_treatment_costs: Optional[Union[float, np.ndarray]] = None
    ) -> SingleTreePolicyInterpreter:
        """
        Attempts to create a SingleTreePolicyInterpreter.

        For all params except 'train',
        see https://econml.azurewebsites.net/_autosummary/econml.cate_interpreter.SingleTreePolicyInterpreter.html
        :param train: false to interpret test set, true to interpret training set, None to interpret full data
        :param include_model_uncertainty:
        :param uncertainty_level:
        :param uncertainty_only_on_leaves:
        :param risk_level:
        :param risk_seeking:
        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param min_weight_fraction_leaf:
        :param max_features:
        :param random_state:
        :param min_impurity_decrease:
        :param min_balancedness_tol:
        :param sample_treatment_costs: sample treatment costs, used in 'interpret' method
        :return:
        """
        ptree: SingleTreePolicyInterpreter = SingleTreePolicyInterpreter(
            include_model_uncertainty=include_model_uncertainty,
            uncertainty_level=uncertainty_level,
            uncertainty_only_on_leaves=uncertainty_only_on_leaves,
            risk_level=risk_level,
            risk_seeking=risk_seeking,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_balancedness_tol=min_balancedness_tol
        )

        return ptree.interpret(
            self.cate_est,
            self.edfm.dfm.x_data(train=train, x_columns=df_utils.DatasetEnum.X_Y),
            sample_treatment_costs=sample_treatment_costs
        )

    def plot_tree(
            self,
            tree: Union[SingleTreeCateInterpreter, SingleTreePolicyInterpreter],
            is_cate: bool,
            max_depth: Optional[int] = None
    ) -> plt.Figure:
        """
        Given either a SingleTreePolicyInterpreter or a SingleTreeCateInterpreter, this attempts to plot that tree.
        :param tree:
        :param is_cate:
        :param max_depth:
        :return: the figure with that tree on it
        """

        tree_type: str = "CATE" if is_cate else "policy"

        fig_ax: Tuple[plt.Figure, plt.Axes] = plt.subplots(
            1,
            1,
            figsize=(16, 9)
        )
        fig: plt.Figure = fig_ax[0]
        ax: plt.Axes = fig_ax[1]

        tree.plot(
            ax=ax,
            title=f"{tree_type} tree for {self.dataset_name} predicted by {self.learner_name}\n",
            feature_names=self.edfm.X.columns.tolist()
        )

        export_relative: str = f"{self.dataset_name}\\{self.learner_name} {tree_type} tree.pdf"

        # noinspection PyBroadException
        try:
            tree.render(
                out_file=f"{os.getcwd()}\\{export_relative}",
                format="pdf",
                max_depth=max_depth
            )
            print(f"Exported via graphviz to {export_relative}!")
        except Exception as e:

            fig.savefig(f"{os.getcwd()}\\{export_relative}")
            print(f"Exported via matplotlib to {export_relative}!")

        return fig

    def save_me(self):

        export_relative: str = f"{self.dataset_name}\\{self.learner_name} CATE.pickle"
        print(f"saving self to {export_relative}!")

        with open(f"{os.getcwd()}\\{export_relative}", "wb") as save_here:
            pickle.dump(self, save_here)
            print("Saved!")

    @classmethod
    def load_me(cls, filename: str) -> "EconMLOutputs":

        with open(filename, "rb") as read_here:
            outputs: "EconMLOutputs" = pickle.load(read_here)
            print("Loaded!")
            return outputs

