import itertools

import more_itertools
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
    Optimize estimator parameters within an interval

    Parameters
    ----------
    estimator : Estimator
        The estimator object. Has to have the `set_params(**params) -> Estimator` and
        `score(data) -> float` methods. Only a single parameter is supported at the
        moment.
    param_bounds : sequence of float
        A sequence containing lower and upper bounds for the parameter.
    optimizer_kwargs : mapping, optional
        Additional parameters for the optimizer
    """

    def __init__(self, estimator, param_bounds, *, optimizer_kwargs={}):
        self.estimator = estimator
        self.param_bounds = tuple(float(v) for v in param_bounds)
        self.optimizer_kwargs = optimizer_kwargs

    def fit(self, X):
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
            result = self.estimator.set_params(sigma=sigma).score(X)
            if not hasattr(result, "compute"):
                return float(result)

            return float(result.compute())

        lower, upper = self.param_bounds
        result = scipy.optimize.fminbound(
            f, lower, upper, args=(X,), **self.optimizer_kwargs
        )

        return self.estimator.set_params(sigma=result.item())

    def fit_final_pos(self, X):
        """Optimize sigma so that predicted final state matches the observed final value."""

        def f(sigma, X):
            result = self.estimator.set_params(sigma=sigma).score_final_pos(X)

            # on suppose que score() met à disposition un dict final_info
            final_info = getattr(self.estimator, "last_info", None)
            if final_info is not None:
                state_val = float(final_info["state_value"])
                final_val = float(final_info["final_value"])
                diff = abs(final_val - state_val)
                print(
                    f"sigma {sigma} | final_value={final_val:.6f} | state_value={state_val:.6f} | diff={diff:.3e}"
                )
                return diff

            if not hasattr(result, "compute"):
                return float(result)
            return float(result.compute())

        lower, upper = self.param_bounds
        result = scipy.optimize.fminbound(
            f, lower, upper, args=(X,), **self.optimizer_kwargs
        )
        return self.estimator.set_params(sigma=result.item())


class TargetBoundsSearch:
    """
    Optimize estimator parameters within an interval

    Parameters
    ----------
    estimator : Estimator
        The estimator object. Has to have the `set_params(**params) -> Estimator` and
        `score(data) -> float` methods. Only a single parameter is supported at the
        moment.
    param_bounds : sequence of float
        A sequence containing lower and upper bounds for the parameter.
    optimizer_kwargs : mapping, optional
        Additional parameters for the optimizer
    """

    def __init__(self, estimator, x0, param_bounds, *, optimizer_kwargs={}):
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.optimizer_kwargs = optimizer_kwargs
        self.x0 = x0

    def fit(self, X):
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

        def f(sigma):
            # computing is important to avoid recomputing as many times as the result is used
            result = self.estimator.set_params(sigma=sigma).score(X)
            if not hasattr(result, "compute"):
                return result

            return result.compute()

        lower, upper = self.param_bounds
        # result = scipy.optimize.fminbound(f, lower, upper, **self.optimizer_kwargs)
        result = scipy.optimize.minimize(
            f,
            self.x0,
            bounds=(0.0, 12.0),  # (lower, upper)
            options=self.optimizer_kwargs,
        )

        return self.estimator.set_params(sigma=result.item())
