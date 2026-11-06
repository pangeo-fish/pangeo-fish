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
    
def detect_segments(predictor_indices):
    """Return (start, end) for each continuous run of identical predictor indices.

    Parameters
    ----------
    predictor_indices : array-like of int
        One predictor index per time step.

    Returns
    -------
    list of tuple of (int, int)
        Half-open (start, end) bounds of each segment.
    """
    predictor_indices = np.asarray(predictor_indices)
    changes = np.where(np.diff(predictor_indices) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(predictor_indices)]])
    return list(zip(starts.tolist(), ends.tolist()))


def group_sigma_indices(predictor_indices, *, single_behav):
    """Build the time indices associated with each sigma.

    Parameters
    ----------
    predictor_indices : array-like of int
        One predictor index per time step.
    single_behav : bool
        If False, one list of time indices per continuous segment.
        If True, one list of time indices per predictor value, sorted by value.

    Returns
    -------
    list of list of int
        The sigma time indices, aligned with the sigma list.
    """
    segments = detect_segments(predictor_indices)

    if not single_behav:
        return [list(range(start, end)) for start, end in segments]

    values = np.asarray(predictor_indices)
    sigma_indices = []
    for value in sorted(set(values.tolist())):
        indices = np.flatnonzero(values == value).tolist()
        sigma_indices.append(indices)
    return sigma_indices

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

    # def __init__(self, estimator, param_bounds, *, optimizer_kwargs={}):
    #     self.estimator = estimator
    #     self.param_bounds = tuple(float(v) for v in param_bounds)
    #     self.optimizer_kwargs = optimizer_kwargs
    #     lower, upper = self.param_bounds

    #     if lower >= upper:
    #         raise ValueError(
    #             "The lower bound must be strictly lower than the upper one."
    #         )

    def __init__(
        self, estimator, param_bounds, optimizer_kwargs=None, sigma_bounds=None
    ):
        """Initialize the bounds searcher.

        Parameters
        ----------
        estimator : object
            The estimator with set_params() and score() methods.
        param_bounds : tuple of (float, float)
            Default bounds (lower, upper) for parameter optimization.
            Used as fallback when sigma_bounds does not specify a group.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for scipy.optimize.fminbound.
        sigma_bounds : dict, optional
            Dynamic-specific bounds as {predictor_index: (lower, upper)}.
            Example: {0: (3, 6), 1: (9, 20)} for two dynamics.
            If None, param_bounds is used for all dynamics.
        """
        self.estimator = estimator
        self.param_bounds = tuple(
            float(v.item() if hasattr(v, "item") else v) for v in param_bounds
        )
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.sigma_bounds = sigma_bounds  # {group_value: (lower, upper)}



    def _get_bounds_for_group(self, group_value):
        """Get optimization bounds for a specific predictor group.

        Parameters
        ----------
        group_value : int or hashable
            The predictor index value.

        Returns
        -------
        tuple of (float, float)
            The (lower, upper) bounds for this group.
            Returns sigma_bounds[group_value] if available,
            otherwise falls back to param_bounds.
        """
        if self.sigma_bounds is not None and group_value in self.sigma_bounds:
            return self.sigma_bounds[group_value]
        return self.param_bounds

    def fit_single_parameter(self, X: xr.Dataset, bounds=None):
        """Optimize a single sigma parameter for a dataset segment.

        Parameters
        ----------
        X : xarray.Dataset
            The input data segment.
        bounds : tuple of (float, float), optional
            Optimization bounds (lower, upper).
            If None, uses self.param_bounds.

        Returns
        -------
        estimator
            The estimator with optimized sigma parameter.
        """
        if bounds is None:
            bounds = self.param_bounds

        def f(sigma, X):
            # computing is important to avoid recomputing as many times as the result is used
            result = self.estimator.set_params(sigma=[sigma]).score(X)
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

        lower, upper = bounds
        result = scipy.optimize.fminbound(
            f, lower, upper, args=(X,), **self.optimizer_kwargs
        )

        return self.estimator.set_params(sigma=[result.item()])

    def fit_multivariate_parameter(
        self,
        X: xr.Dataset,
        group_name: str = "predictor_index",
        x0=None,
        method: str = "Powell",
        maxfev: int = 20,
        disp: int = 1,
    ):
        """Optimize one sigma per predictor value jointly on the whole sequence.

        All sigmas are optimized at once with ``scipy.optimize.minimize``. The
        objective is the total negative log-likelihood of the sequence, where each
        transition uses the sigma of its predictor index and states are chained
        across segments.

        .. note::
            ``single_behav`` has no effect in this mode: the number of sigmas is
            always the number of predictor values.

        Parameters
        ----------
        X : xarray.Dataset
            Dataset with ``pdf``, ``mask``, ``initial`` and ``group_name``.
        group_name : str, default: "predictor_index"
            Name of the variable holding the predictor index per time step.
        x0 : sequence of float, optional
            Starting sigmas, one per predictor value (sorted). Defaults to the
            middle of each bound.
        method : str, default: "Powell"
            Optimization method passed to ``scipy.optimize.minimize``. Use a
            method that does not require gradients.
        maxfev : int, default: 20
            Maximum number of evaluations of the objective.
        disp : int, default: 0
            Verbosity passed to ``scipy.optimize.minimize``.

        Returns
        -------
        estimator
            The estimator with the optimized sigmas, ordered by predictor value.
        sigma_indices : list of list of int
            Time indices associated with each sigma, aligned with ``sigma``.
        """
        if group_name not in X:
            raise ValueError(f'The dataset must have an entry "{group_name}".')

        predictor_indices = X[group_name].values
        sigma_indices = group_sigma_indices(predictor_indices, single_behav=True)

        unique_values = sorted(set(predictor_indices.tolist()))
        bounds = [self.sigma_bounds[v] for v in unique_values]
        if x0 is None:
            x0 = [np.mean(b) for b in bounds]

        estimator = self.estimator
        if method == "Powell":
            options={'xtol': self.optimizer_kwargs["xtol"], "maxfev": maxfev, "disp": bool(disp) }
        if method == "Nelder-Mead":
            options={'xatol': self.optimizer_kwargs["xtol"], "maxfev": maxfev, "disp": bool(disp) }

        history = []
        def f(sigmas):
            result = estimator.set_params(sigma=list(sigmas)).score(X)
            value = float(result.compute() if hasattr(result, "compute") else result)
            print(f"sigmas = {sigmas}    score = {result}")
            history.append((np.asarray(sigmas, dtype=float), value))

            return value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = scipy.optimize.minimize(
                f,
                x0=x0,
                bounds=bounds,
                method=method,
                options=options,
            )
        best_sigmas, best_score = min(history, key=lambda h: h[1])

        self.multivariate_result = result
        self.multivariate_history = history
        return estimator.set_params(sigma=best_sigmas.tolist()), sigma_indices



    def fit_multiple_single_parameters(
        self, X: xr.Dataset, group_name: str = "predictor_index", single_behav: bool = True,
    ):
        """Optimize parameters by continuous temporal segments.

        Each continuous segment with the same predictor index is optimized
        independently using its group-specific bounds. State is chained
        between segments to maintain temporal continuity.

        .. note::
            Unlike the previous approach, non-consecutive indices
            (e.g., [0,0,0,1,1,1,0,0,0]) are now treated as separate
            segments, avoiding temporal discontinuities during optimization.

        Parameters
        ----------
        X : xarray.Dataset
            The input data with "predictor_index" variable and "initial" condition.
        group_name : str, default: "predictor_index"
            Name of the variable defining predictor groups.
        single_behav : bool, default: True
            Define how behavior is treated (one behavior per values or one per segments)
        Returns
        -------
        estimator
            The unified estimator with all optimized parameters.
        segments
            The list of indices per segments
        Raises
        ------
        ValueError
            If the dataset does not contain the group_name variable.
        """

        if group_name not in X:
            raise ValueError(f'The dataset must have an entry "{group_name}".')

        # Detect continuous segments
        predictor_indices = X[group_name].values
        segments = detect_segments(predictor_indices)
        if self.sigma_bounds is not None and len(self.sigma_bounds) != len(
            set(predictor_indices)
        ):
            raise ValueError(
                f"The dict for behavior speed:{len(self.sigma_bounds)} should have the same length that the number predictor_indices:{len(set(predictor_indices))}."
            )
        if not single_behav:
            # One sigma per continuous segment, each optimized independently

            estimators = []
            previous_initial = X["initial"]

            # Optimize each segment sequentially
            for start, end in segments:
                # Extract the segment
                subset = X.isel(time=slice(start, end))
                group_value = predictor_indices[start]

                # Pass the initial state from the previous segment
                subset = subset.assign(
                    initial=previous_initial,
                    predictor_index=(
                        "time",
                        np.zeros(subset["time"].size).astype(np.int32),
                    ),
                )
                # Get bounds for this group
                bounds = self._get_bounds_for_group(group_value)

                # Optimize this segment's parameter
                estimated = self.fit_single_parameter(subset, bounds=bounds)
                estimators.append(estimated)

                # Update initial state for next segment using final state of current segment
                previous_initial = subset["pdf"].isel(time=-1).fillna(0)

            # Aggregate all optimized parameters
            estimated_params = [e.sigma for e in estimators]
            return (self.estimator.set_params(sigma=sum(estimated_params, start=[])),[list(range(start, end)) for start, end in segments],)

        # One sigma per predictor value, optimized on the summed log-likelihood
        # of all segments sharing that value. Each segment is initialized with
        # the observed pdf at its first time step.
        unique_values = sorted(set(predictor_indices[start] for start, _ in segments))
        print(unique_values)
        sigmas = []
        group_indices = []

        for group_value in unique_values:
            group_segments = [
                (start, end)
                for start, end in segments
                if predictor_indices[start] == group_value
            ]
            bounds = self._get_bounds_for_group(group_value)

            def total_score(sigma, group_segments=group_segments):
                total = 0.0
                for start, end in group_segments:
                    subset = X.isel(time=slice(start, end))
                    subset = subset.assign(
                        initial=X["initial"] if start == 0 else X["pdf"].isel(time=start - 1).fillna(0),
                        predictor_index=("time", np.zeros(subset["time"].size).astype(np.int32)),
                    )
                    result = self.estimator.set_params(sigma=[sigma]).score(subset)
                    total += float(result.compute() if hasattr(result, "compute") else result)
                return total

            lower, upper = bounds
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                best = scipy.optimize.fminbound(
                    total_score, lower, upper, **self.optimizer_kwargs
                )

            sigmas.append(float(best))
            group_indices.append(
                [i for start, end in group_segments for i in range(start, end)]
            )
        return (
            self.estimator.set_params(sigma=sigmas),
            group_indices,
            )


    def fit_final_pos(self, X):
        """Optimize sigma so that predicted final state matches the observed final value."""

        def f(sigma, X):
            result = self.estimator.set_params(sigma=[sigma]).score_final_pos(X)

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
        return self.estimator.set_params(sigma=[result.item()])


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
            result = self.estimator.set_params(sigma=sigma).score(X)
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
            sigma = [opt_result.x.item()]
        else:
            warnings.warn(
                "Minimization failed. Estimator's parameter set to None.",
                RuntimeWarning,
            )
            sigma = None

        return self.estimator.set_params(sigma=sigma)
