"""Define _TSP Models

Time Series Predictions (TSPs) are attempt to predict what will happen based
on what has happened before. While there are a plethora of ways to do this,
the teaspoon module foucsses on using the last few observations to predict
the next and mechanisms to combine several of these predictions.
"""

import multiprocessing

import pandas as pd
import numpy as np


def ts_to_labels(_ts, _n, col=None):
    """Convert a time series iterable into a set of features and labels ready
    for training.

    Args:
        _ts (array-like): time series to be used for training.
        _n (int): number of step features for each label.
        col (any): column identifier for dataframe time series, in case only
            a subsection of it will be used for training.
    """
    _ts = _ts if isinstance(_ts, pd.DataFrame) \
        else pd.DataFrame(_ts, columns=["x"])

    _x, _y = list(), list()
    _ts.rolling(_n+1).apply(append_window,
                            args=(_x, _y, _n),
                            kwargs={"col": col})

    return np.array(_x), np.array(_y)


def append_window(_w, _x, _y, _n, col=None):
    """Helper function to append the features and labels from a time series
    rolling window into a feature and label array.

    Args:
        _w (pd.DataFrame or pd.Series): time series data window element of
            the .rolling(_n+1) method.
        _x (list): feature list to append features to.
        _y (list): feature list to append features to.
        _n (int): number of step features for each label.
        col (any): column identifier for dataframe time series, in case only
            a subsection of it will be used for training.
    """
    _x.append(np.array(_w.iloc[:_n]))
    _y.append(np.array(_w.iloc[_n]) if col is None
              else np.array(_w.iloc[_n][col]))

    return 1


class SimpleModelWrapper:
    """Wrapper object used to "translate" a model's core functionaliy into
    one that can be used in _TSP instances.

    This wrapper by default simply calls an alternative function as specifed
    upon initialization, with assumed positional arguments.

    This class can be inheritted to incorporate more complex mechanics of
    whichever model is being used.

    Attributes:
        _model (any): model with fit and predict capabilities.
        _fit_attr (str): model attribute used for fitting.
        _predict_attr (str): model attribute used for predicting values.
    """

    def __init__(self, model, fit_attr="fit", predict_attr="predict"):
        """Initialize object instance.

        Args:
            model (any): model with fit and predict capabilities.
            fit_attr (str), default "fit": model attribute used for fitting.
            predict_attr (str), default "predict": model attribute used for
                predicting values.

        Raise:
            TypeError: if fit_attr or predict_attr are not strings.
        """
        self._model = model

        if not isinstance(fit_attr, str):
            raise TypeError(f"fit_attr parameter must be {str}, \
                not {type(fit_attr)}")

        self._fit_attr = fit_attr

        if not isinstance(predict_attr, str):
            raise TypeError(f"predict_attr parameter must be {str}, \
                not {type(predict_attr)}")

        self._predict_attr = predict_attr

    def fit(self, features, labels, *args, **kwargs):
        """Fit model(s)

        Args:
            features (array or matrix like): features used for fitting
            labels (array or matrix like): labels used for fitting
            *args, **kwargs: arguments used for fitting
        """
        return self._model.__getattribute__(self._fit_attr)(
            features,
            labels,
            *args,
            **kwargs
        )

    def predict(self, features, *args, **kwargs):
        """Predict value(s) 

        Args:
            features (array or matrix like): features used for fitting
            *args, **kwargs: arguments used for fitting
        """
        return self._model.__getattribute__(self._predict_attr)(
            features,
            *args,
            **kwargs
        )


class _TSP:
    """Abstract Time Series Prediction class.

    Attributes:
        _model (any with "fit" and "predict" parameter): model that takes in
            past steps and predicts future ones.
    """

    def fit(self, _ts, *args, **kwargs):
        """Fit model from data.

        Args:
            _ts (array-like): time series data used to fit the model.
        """
        raise NotImplementedError()

    def predict(self, _ts, *args, start=None, horizon=1, **kwargs):
        """Predict future steps from past ones.

        Args:
            _ts (array-like): time series to get past steps from.
            start (any), Optional, None by default: first step to predict
                from.
            horizon (int), 1 by default: how many steps ahead to predict.
        """
        raise NotImplementedError()

    @property
    def model(self):
        """Define model fetching mechanism to ensure model must be set to
        be accessed

        Raises:
            AttributeError: if model is not set.
        """

        if self._model is None:
            raise AttributeError("model attribute is not set")

        return self._model

    @model.setter
    def model(self, new_model):
        """Define model setting mechanism to ensure model can be fit and
        used for prediction.

        Raises:
            AttributeError: if model does not have a "fit" or "predict"
                parameter.
        """

        if not hasattr(new_model, "fit"):
            raise AttributeError("specified model must have a 'fit' \
                attribute")

        if not hasattr(new_model, "predict"):
            raise AttributeError("specified model must have a 'predict' \
                attribute")

        self._model = new_model


class UTSP(_TSP):
    """Univarate Time Series Prediction model.

    This is used to predict the next step given a one-dimentional array.

    Attributes:
        _n (int): how many past steps considered for predicting the next.
    """

    def __init__(self, model, n):
        """Initialize model parameters.

        Args:
            n (int): how many past steps considered for predicting the next.
            model (any): fittable model that takes in {n} one-dimentional
                inputs and returns a single value for the predicted next
                step.
        """

        self.model = model

        self._n = n

    def fit(self, _ts, *args, shuffle=True, **kwargs):

        if (len(_ts.shape) == 1 and _ts.shape[1] != 1) \
                or (len(_ts.shape) >= 2):

            raise ValueError(f"input time series must be a 1D array, not \
                {len(_ts.shape)}D")

        _x, _y = ts_to_labels(_ts, self._n)
        _x = _x.reshape(-1, 1) if len(_x.shape) == 1 else _x
        _y = _y.reshape(-1, 1)

        if shuffle:
            concat = np.concatenate((_x, _y), axis=1)
            np.random.shuffle(concat)
            _x, _y = np.split(concat, [self._n], axis=1)

        self.model.fit(_x, _y, *args, **kwargs)

    def predict(self, _ts, *args, start=None, horizon=1, **kwargs):

        if (len(_ts.shape) == 1 and _ts.shape[1] != 1) \
                or (len(_ts.shape) >= 2):
            raise ValueError(f"input time series must be a 1D array, not \
                {len(_ts.shape)}D")

        if len(_ts) < self._n:
            ValueError(f"input musut have at least {self._n} items.")

        ret_x, ret_pred = [], []

        curr_x, curr_pred = np.empty(0), None
        if start is None:
            curr_x = _ts[:-self._n]
        else:
            if len(_ts[start:]) < self._n:
                ValueError(f"specify a start with more than {self._n} items \
                    ahead of it.")
            curr_x = _ts[start:start+self._n]

        curr_x = np.array(curr_x)

        for _ in range(horizon):
            curr_pred = self.model.predict(np.array(curr_x.reshape(1, -1)),
                                           *args,
                                           **kwargs)
            ret_x.append(curr_x)
            ret_pred.append(curr_pred)
            curr_x[:self._n-1], curr_x[self._n-1] = curr_x[1:], curr_pred

        return np.array(ret_x), np.array(ret_pred)


class MTSP(_TSP):
    """Multivariate Time Series Prediction models.

    These models are highly flexible ways of predicting future values based 
    on a 2D aray of last steps over multiple features. While this feature is
    commonly used to look a single step ahead, users can specify more 
    granularly how each step variable on a step ahead should be predicted.

    Attributes:
        _n (int): number of steps before used to predict a step ahead.
        _col (str): column this model tried to predict.
        _submodels (dict): dictionary of submodels used to predict a column's
            value based on previous steps. These must be either USTP
            models for single columns or MTSP for multiple columns. This 
            dictionary should have a tuple (multiple columns) or string 
            (single column) as keys and _TSP instances as values. 
        _min_cols (list): all columns specified by the initialization.
        n_jobs (int): number of jobs to run for fiting and predicting.
    """

    def __init__(self,
                 model,
                 n,
                 col,
                 submodels=None,
                 n_jobs=1):
        """

        Args:
            n (int): number of steps before used to predict a step ahead.
            col (str): column this model tried to predict.
            submodels (dict): dictionary of submodels used to predict any 
                dataframe varible in a custom way. These must be either USTP
                models for single columns or MTSP for multiple columns. This 
                dictionary should have a tuple (multiple columns) or string 
                (single column) as keys and _TSP instances as values. This 
                variable will be filled upon fitting to account for 
                unspecified columns.
            n_jobs (int): number of jobs to run for fiting and predicting.
        """

        self.model = model

        min_cols = set()

        self._col = col
        min_cols.add(self._col)

        self._n = n

        if isinstance(submodels, dict):
            raise TypeError(f"mutlistep_models parameter must be of type \
                {dict} not {type(submodels)}")

        for col_name, tsp in submodels.items():
            if isinstance(col_name, tuple):

                if not isinstance(tsp, MTSP):
                    raise TypeError(f"multistep model for column {col_name} \
                        must be of type {MTSP} not {type(tsp)} if \
                        predicting based on single variable")

                col1, col2 = col

                min_cols.add(col1)

                if isinstance(col2, (tuple, list)):
                    min_cols.update(col2)
                else:
                    min_cols.add(col2)

            else:

                if not isinstance(tsp, UTSP):
                    raise TypeError(f"multistep model for column {col_name} \
                        must be of type {UTSP} not {type(tsp)} if \
                        predicting based on multiple variables")

                min_cols.add(col)

        self._min_cols = list(min_cols)

        self.n_jobs = n_jobs

    def fit(self, _ts, *args, **kwargs):

        if not isinstance(_ts, pd.DataFrame):
            raise TypeError(f"argument _ts must be of type {pd.DataFrame} \
                not {type(_ts)}")

        if not all([col_name in _ts.columns
                    for col_name in self._min_cols]):
            raise ValueError(f"time series should have the following columns \
                    specified upon model initialization: {self._min_cols}")

        _x, _y = ts_to_labels(_ts, self._n)

        self.model.fit(_x,
                       _y.reshape(-1, 1),
                       *args,
                       **kwargs)

        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = [pool.apply_async(tsp.fit, (_ts[col_name],))
                       for col_name, tsp in self._submodels]

            for res in results:
                res.get()

    def predict(self, _ts, *args, start=None, horizon=1, **kwargs):

        if not isinstance(_ts, pd.DataFrame):
            raise TypeError(f"argument _ts must be of type {pd.DataFrame} \
                not {type(_ts)}")

        if len(_ts) < self._n:
            ValueError(f"input musut have at least {self._n} items.")

        if not all([col_name in _ts.columns
                    for col_name in self._min_cols]):
            raise ValueError(f"time series should have the following columns \
                    specified upon model initialization: {self._min_cols}")

        ret_x, ret_pred = [], []

        curr_x, curr_pred = np.empty(0), None
        if start is None:
            start = len(_ts) - self._n

        if len(_ts.iloc[start:]) < self._n:
            ValueError(f"specify a start with more than {self._n} items \
                ahead of it.")

        # we will append to the time series, so we create a copy now.
        # copy will only have the necessary number of steps behind to 
        # save memory.
        _ts = _ts.copy().iloc[start:
                              -max([sm._n for sm in self._submodels.values()]
                                   + [self._n])]

        curr_x = _ts.iloc[start:start+self._n].values
        col_names_idx = {col: i for i, col in enumerate(_ts.columns)}

        pred_cols = col_names_idx[self._col] if isinstance(self._col, str) \
            else [col_names_idx[c] for c in self._col]

        for _ in range(horizon):
            curr_pred = self.model.predict(curr_x.reshape(1, -1),
                                           *args,
                                           **kwargs)
            ret_x.append(curr_x)
            ret_pred.append(curr_pred)

            new_step = curr_x[-1]

            new_step[pred_cols] = curr_pred

            for col_name, tsp in self._submodels:
                # TODO: parallelize
                if isinstance(col_name, tuple):
                    col_name, col_sl = col_name
                else:
                    col_sl = col_name

                new_step[col_names_idx[col_name]] = tsp.predict(_ts[col_sl])

            curr_x[:self._n-1], curr_x[-1] = curr_x[1:], new_step
            _ts[len(_ts)] = new_step

        return np.array(ret_x), np.array(ret_pred)

    @property
    def n_jobs(self):
        """Get n_jobs attribute"""

    @n_jobs.setter
    def n_jobs(self, _n):
        """Set n_jobs attribute ensuring it new value is an integer"""

        if not isinstance(_n, int):
            raise TypeError(f"attribute n_jobs must be of type {int} \
                not {type(_n)}")

        self._n_jobs = _n
