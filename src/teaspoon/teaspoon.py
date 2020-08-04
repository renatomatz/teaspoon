"""Define _TSP Models

Time Series Predictions (TSP).
"""

import multiprocessing

import pandas as pd
import numpy as np


def ts_to_labels(_ts, _n, col=None):
    """
    Args:
        _ts ():
        _n ():
        col ():
    """
    _ts = _ts if isinstance(_ts, pd.DataFrame) \
        else pd.DataFrame(_ts, columns="x")

    _x, _y = list(), list()
    _ts.rolling(_n+1).apply(append_window,
                            args=(_x, _y, _n),
                            kwargs={"col": col})

    return np.array(_x), np.array(_y)


def append_window(_w, _x, _y, _n, col=None):
    """
    Args:
        _w ():
        _x ():
        _y ():
        _n ():
        col ():
    """
    _x.append(np.array(_w.iloc[:_n]))
    _y.append(np.array(_w.iloc[_n]) if col is None
              else np.array(_w.iloc[_n][col]))
    return 1


class _TSP:
    """
    """

    def fit(self, _ts):
        """
        Args:
            _ts ():
        """
        raise NotImplementedError()

    def predict(self, _ts, start=None, horizon=1):
        """
        Args:
            _ts ():
            start ():
            horizon ():
        """
        raise NotImplementedError()

    @property
    def model(self):
        """Define model fetching mechanism to ensure model must be set to
        be accessed
        """

        if self._model is None:
            raise AttributeError("model attribute is not set")

        return self._model

    @model.setter
    def model(self, new_model):
        """Define model setting mechanism to ensure model can be fit"""

        if new_model is not None and not hasattr(new_model, "fit"):
            raise AttributeError("specified model must have a 'fit' \
                attribute")

        self._model = new_model


class UTSP(_TSP):
    """
    Attributes:
        _n ():
    """

    def __init__(self, n, model=None):
        """
        Args:
            n ():
            model ():
        """

        self.model = model

        self._n = n

    def fit(self, _ts):

        if len(_ts.shape) != 1:
            raise ValueError(f"input time series must be a 1D array, not \
                {len(_ts.shape)}D")

        _x, _y = ts_to_labels(_ts, self._n)

        self.model.fit(_x.reshape(-1, 1) if len(_x.shape) == 1 else _x,
                       _y.reshape(-1, 1))

    def predict(self, _ts, start=None, horizon=1):

        if len(_ts.shape) != 1:
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
            curr_pred = self.model.predict(np.array(curr_x.reshape(1, -1)))
            ret_x.append(curr_x)
            ret_pred.append(curr_pred)
            curr_x[:self._n-1], curr_x[self._n-1] = curr_x[1:], curr_pred

        return np.array(ret_x), np.array(ret_pred)


class MTSP(_TSP):
    """
    Attributes:
        _col ():
        _n ():
        _submodels ():
        _n_processes (int):
        _all_cols ():
    """

    def __init__(self,
                 col,
                 n,
                 model=None,
                 submodels=None,
                 n_processes=1):
        """
        Args:
            col ():
            n ():
            model ();
            submodels ():
            n_processes (int):
        """

        self.model = model

        all_cols = set()

        self._col = col
        all_cols.add(self._col)

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

                all_cols.add(col1)

                if isinstance(col2, (tuple, list)):
                    all_cols.update(col2)
                else:
                    all_cols.add(col2)

            else:

                if not isinstance(tsp, UTSP):
                    raise TypeError(f"multistep model for column {col_name} \
                        must be of type {UTSP} not {type(tsp)} if \
                        predicting based on multiple variables")

                all_cols.add(col)

        self._submodels = submodels \
            if submodels is not None else dict()

        self._all_cols = list(all_cols)

        self.n_processes = n_processes

    def fit(self, ts):

        if not isinstance(ts, pd.DataFrame):
            raise TypeError(f"argument ts must be of type {pd.DataFrame} \
                not {type(ts)}")

        if not all([col_name in ts.columns
                    for col_name in self._all_cols]):
            raise ValueError(f"time series should have the following columns \
                    specified upon model initialization: {self._all_cols}")

        _x, _y = ts_to_labels(ts, self._n)

        self.model.fit(_x,
                       _y.reshape(-1, 1))

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = [pool.apply_async(tsp.fit, (ts[col_name],))
                       for col_name, tsp in self._submodels]

            for res in results:
                res.get()

    def predict(self, ts, start=None, horizon=1):

        if not isinstance(ts, pd.DataFrame):
            raise TypeError(f"argument ts must be of type {pd.DataFrame} \
                not {type(ts)}")

        if len(ts) < self._n:
            ValueError(f"input musut have at least {self._n} items.")

        if not all([col_name in ts.columns
                    for col_name in self._all_cols]):
            raise ValueError(f"time series should have the following columns \
                    specified upon model initialization: {self._all_cols}")

        ret_x, ret_pred = [], []

        curr_x, curr_pred = np.empty(0), None
        if start is None:
            start = len(ts) - self._n

        if len(ts.iloc[start:]) < self._n:
            ValueError(f"specify a start with more than {self._n} items \
                ahead of it.")

        curr_x = ts.iloc[start:start+self._n].values
        col_names_idx = {col: i for i, col in enumerate(ts.columns)}

        for _ in range(horizon):
            curr_pred = self.model.predict(curr_x.reshape(1, -1))
            ret_x.append(curr_x)
            ret_pred.append(curr_pred)

            new_step = curr_x[-1]

            ts_cp = ts.copy()
            new_step[col_names_idx[self._col]] = curr_pred

            for col_name, tsp in self._submodels:
                # TODO: parallelize
                if isinstance(col_name, tuple):
                    col_name, col_sl = col_name
                else:
                    col_sl = col_name

                new_step[col_names_idx[col_name]] = tsp.predict(ts_cp[col_sl])

            curr_x[:self._n-1], curr_x[-1] = curr_x[1:], new_step
            ts_cp[len(ts_cp)] = new_step

        return np.array(ret_x), np.array(ret_pred)

    @property
    def n_processes(self):
        """Get n_processes attribute"""
        return self._n_processes

    @n_processes.setter
    def n_processes(self, _n):
        """Set n_processes attribute ensuring it new value is an integer"""

        if not isinstance(_n, int):
            raise TypeError(f"attribute n_processes must be of type {int} \
                not {type(_n)}")

        self._n_processes = _n
