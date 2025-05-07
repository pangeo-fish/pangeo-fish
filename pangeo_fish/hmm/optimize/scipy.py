import itertools
import warnings

import more_itertools
import numpy as np
import scipy.optimize
import xarray as xr

try:
    from rich.progress import track
except ImportError:

    def track(iterable, **kwargs):
        return iterable


class GridSearch:
    """
    Optimize estimator parameters using a search grid

    Parameters
    ----------
    estimator : Estimator
        The estimator object. Has to have the `set_params(**params) -> Estimator` and
        `score(data) -> float` methods. Only a single parameter is supported at the
        moment.
    search_grid : mapping of str to sequence
        The search grid.
    """

    def __init__(self, estimator, search_grid):
        self.estimator = estimator
        self.search_grid = search_grid

    def fit(self, X):
        """
        search for optimal parameters

        Parameters
        ----------
        X : xarray.Dataset
            The input data.

        Returns
        -------
        optimized : Estimator
            The estimator with optimized parameters.
        """
        grid_items = [
            [{name: value} for value in values]
            for name, values in self.search_grid.items()
        ]
        trials = [
            dict(itertools.chain.from_iterable(item.items() for item in items))
            for items in itertools.product(*grid_items)
        ]
        results = [
            self.estimator.set_params(**params).score(X)
            for params in track(trials, description="Creating task graph...")
        ]
        combined = xr.combine_by_coords(
            [
                result.assign_coords(params).expand_dims(list(params))
                for params, result in zip(trials, results)
            ]
        )
        optimized_params = combined.idxmin().to_array(dim="variable").compute().item()

        return self.estimator.set_params(
            **{more_itertools.first(self.search_grid): optimized_params}
        )


class EagerBoundsSearch:
    """
    Class for optimizing the parameters of an Estimator within an interval.

    Parameters
    ----------
    estimator : Estimator
        The estimator object. Has to have the `set_params(**params) -> Estimator` and
        `score(data) -> float` methods. Each parameter is git optimized individually.
    param_bounds : sequence of float
        A sequence containing lower and upper bounds for the parameters.
    optimizer_kwargs : mapping, optional
        Additional parameters for the optimizer.
    """

    def __init__(self, estimator, param_bounds, *, optimizer_kwargs={}):
        self.estimator = estimator
        self.param_bounds = tuple(float(v) for v in param_bounds)
        self.optimizer_kwargs = optimizer_kwargs

        lower, upper = self.param_bounds

        if lower >= upper:
            raise ValueError(
                "The lower bound must be strictly lower than the upper one."
            )

    def fit_single_parameter(self, X: xr.Dataset):
        """Optimize the score of the estimator with a single parameter.

        Parameters
        ----------
        X : xarray.Dataset
            The input data.

        Returns
        -------
        estimator
            The estimator with optimized parameters.
        """

        def f(sigma, X):
            # computing is important to avoid recomputing as many times as the result is used
            result = self.estimator.set_params(sigmas=[sigma]).score(X)
            if not hasattr(result, "compute"):
                return float(result)

            return float(result.compute())

        if "predictor_index" in X:
            if X["predictor_index"].dtype != np.int32:
                X["predictor_index"] = X["predictor_index"].astype(np.int32)
        else:
            X = X.assign(
                predictor_index=("time", np.zeros(X["time"].size).astype(np.int32))
            )

        lower, upper = self.param_bounds
        result = scipy.optimize.fminbound(
            f, lower, upper, args=(X,), **self.optimizer_kwargs
        )

        return self.estimator.set_params(sigmas=[result.item()])

    def fit_multivariate_parameter(self, X: xr.Dataset):
        """
        .. warning::
            Not implemented yet.
        """
        raise NotImplementedError()

    def fit_multiple_single_parameters(
        self, X: xr.Dataset, group_name: str = "predictor_index"
    ):  # , time_slice_indices: list[int | None]):
        """Optimize the score of the estimator.

        .. warning::
            Each parameter ``sigma`` is individually optimized with a subset of ``X`` defined by the entry ``X.groupby(group_name)``.
            As such, it is **not** a single, multivariate search.

        Parameters
        ----------
        X : xarray.Dataset
            The input data.
        group_name : str, default: "predictor_index"
            Name of the variable for groupying ``X``. For each group, a single-parameter Estimator is optimized.

        Returns
        -------
        estimator
            The unified estimator with all the optimized parameters found for the groups.

        See Also
        --------
        EagerBoundsSearch.fit_multivariate_parameter
        """

        if group_name not in X:
            raise ValueError(f'The dataset must have an entry "{group_name}".')

        estimators = []

        # for i in range(len(time_slice_indices) - 1):
        for i, (group_value, group_ds) in enumerate(X.groupby(group_name)):
            # subset = (
            #     X.isel(time=slice(time_slice_indices[i], time_slice_indices[i + 1]))
            #     .drop_vars("initial", errors="ignore")
            # )
            subset = group_ds.drop_vars(["initial", group_name], errors="ignore")
            estimated = self.fit_single_parameter(
                subset.assign(
                    initial=(
                        X["initial"] if i == 0 else subset["pdf"].isel(time=0).fillna(0)
                    ),
                    # it is important to replace (or set) the current predictor's index by 0
                    predictor_index=(
                        "time",
                        np.zeros(subset["time"].size).astype(np.int32),
                    ),
                )
            )
            estimators.append(estimated)

        estimated_params = [e.sigmas for e in estimators]
        return self.estimator.set_params(sigmas=sum(estimated_params, start=[]))


class TargetBoundsSearch:
    """
    Class for minimazing the parameter of an Estimator within an interval.

    Parameters
    ----------
    estimator : Estimator
        The estimator object. Has to have the `set_params(**params) -> Estimator` and
        `score(data) -> float` methods. A single parameter is currently supported.
    x0 : float
        The initial value to start the minimization.
    param_bounds : sequence of float
        A sequence containing lower and upper bounds for the parameters.
    optimizer_kwargs : mapping, optional
        Additional parameters for the optimizer.
    """

    def __init__(
        self, estimator, x0: float, param_bounds, *, optimizer_kwargs: dict = {}
    ):

        self.estimator = estimator
        self.param_bounds = tuple(float(v) for v in param_bounds)
        self.optimizer_kwargs = optimizer_kwargs
        self.x0 = x0

        lower, upper = self.param_bounds

        if lower >= upper:
            raise ValueError(
                "The lower bound must be strictly lower than the upper one."
            )

        if (x0 < lower) or (x0 > upper):
            raise ValueError("The initial value `x0` must be within the bounds.")

    def fit(self, X: xr.Dataset):
        """Optimize the score of the estimator

        Parameters
        ----------
        X : xarray.Dataset
            The input data.

        Returns
        -------
        estimator
            The estimator with optimized parameters.
        """

        def f(sigma, X):
            # computing is important to avoid recomputing as many times as the result is used
            result = self.estimator.set_params(sigmas=sigma).score(X)
            if not hasattr(result, "compute"):
                return float(result)

            return float(result.compute())

        X = X.assign(
            predictor_index=("time", np.zeros(X["time"].size).astype(np.int32))
        )

        tol = self.optimizer_kwargs.pop("tol", None)

        opt_result = scipy.optimize.minimize(
            f,
            x0=(self.x0,),
            args=(X,),
            bounds=[self.param_bounds],
            tol=tol,
            options=self.optimizer_kwargs,
        )

        if opt_result.success:
            sigmas = [opt_result.x.item()]
        else:
            warnings.warn(
                "Minimization failed. Estimator's parameter set to None.",
                RuntimeWarning,
            )
            sigmas = None

        return self.estimator.set_params(sigmas=sigmas)
