from dataclasses import asdict, dataclass, replace

import movingpandas as mpd
import numpy as np
import xarray as xr
from tlz.functoolz import compose_left, curry, pipe
from tlz.itertoolz import first

from ... import utils
from ...tracks import to_trajectory
from ..decode import mean_track, modal_track, viterbi, viterbi2
from ..filter import forward, forward_backward, score


@dataclass
class EagerScoreEstimator:
    """Estimator to train and predict gaussian random walk hidden markov models

    Parameters
    ----------
    sigma : float, default: None
        The primary model parameter: the standard deviation of the distance
        per time unit traveled by the fish, in the same unit as the grid coordinates.
    truncate : float, default: 4.0
        The cut-off limit of the filter. This can be used, together with `sigma`, to
        calculate the maximum distance per time unit traveled by the fish.
    """

    sigma: float = None
    truncate: float = 4.0

    def to_dict(self):
        return asdict(self)

    def set_params(self, **params):
        """set the parameters on a new instance

        Parameters
        ----------
        **params
            Mapping of parameter name to new value.
        """
        return replace(self, **params)

    def _score(self, X, *, spatial_dims=None, temporal_dims=None):
        import gc

        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, *, sigma, truncate):
            return score(
                emission,
                mask=mask,
                initial_probability=initial,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
        ]

        value = xr.apply_ufunc(
            _algorithm,
            X.pdf.fillna(0),
            X.mask,
            X.initial,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[()],
            dask="allowed",
        )
        gc.collect()
        return value.fillna(np.inf)

    def _forward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, *, sigma, truncate):
            return forward(
                emission=emission,
                mask=mask,
                initial_probability=initial,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[temporal_dims + spatial_dims],
            dask="allowed",
        )

    def _backward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, *, sigma, truncate):
            pass

    def _forward_backward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, *, sigma, truncate):
            backward_state = forward_backward(
                emission=emission,
                sigma=sigma,
                mask=mask,
                initial_probability=initial,
                truncate=truncate,
            )

            return backward_state

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[temporal_dims + spatial_dims],
            dask="allowed",
        )

    def predict_proba(self, X, *, spatial_dims=None, temporal_dims=None):
        """predict the state probabilities

        This is done by applying the forward-backward algorithm to the data.

        Parameters
        ----------
        X : Dataset
            The emission probability maps. The dataset should contain these variables:
            - `initial`, the initial probability map
            - `pdf`, the emission probabilities
            - `mask`, a mask to select ocean pixels
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Returns
        -------
        state_probabilities : DataArray
            The computed state probabilities
        """
        state = self._forward_backward_algorithm(
            X.fillna(0), spatial_dims=spatial_dims, temporal_dims=temporal_dims
        )
        return state.rename("states")

    def score(self, X, *, spatial_dims=None, temporal_dims=None):
        """score the fit of the selected model to the data

        Apply the forward-backward algorithm to the given data, then return the
        negative logarithm of the normalization factors.

        Parameters
        ----------
        X : Dataset
            The emission probability maps. The dataset should contain these variables:
            - `pdf`, the emission probabilities
            - `mask`, a mask to select ocean pixels
            - `initial`, the initial probability map
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Return
        ------
        score : float
            The score for the fit with the current parameters.
        """
        return self._score(
            X.fillna(0), spatial_dims=spatial_dims, temporal_dims=temporal_dims
        )

    def decode(
        self,
        X,
        states=None,
        *,
        mode="viterbi",
        spatial_dims=None,
        temporal_dims=None,
        progress=False,
    ):
        """decode the state sequence from the selected model and the data

        Parameters
        ----------
        X : Dataset
            The emission probability maps. The dataset should contain these variables:
            - `pdf`, the emission probabilities
            - `mask`, a mask to select ocean pixels
            - `initial`, the initial probability map
            - `final`, the final probability map (optional)
        states : Dataset, optional
            The precomputed state probability maps. The dataset should contain these variables:
            - `states`, the state probabilities
        mode : {"mean", "mode", "viterbi"}, default: "viterbi"
            The decoding method. Can be one of
            - ``"mean"``: use the centroid of the state probabilities as decoded state
            - ``"mode"``: use the maximum of the state probabilities as decoded state
            - ``"viterbi"``: use the viterbi algorithm to determine the most probable states
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.
        """

        def maybe_compute_states(data):
            X, states = data

            if states is None:
                return self.predict_proba(
                    X, spatial_dims=spatial_dims, temporal_dims=temporal_dims
                )

            return states

        decoders = {
            "mean": compose_left(maybe_compute_states, mean_track),
            "mode": compose_left(maybe_compute_states, modal_track),
            "viterbi": compose_left(first, curry(viterbi, sigma=self.sigma)),
            "viterbi2": compose_left(first, curry(viterbi2, sigma=self.sigma)),
        }

        # check modes available
        # decode
        # to_trajectory
        # convert to TrajectoryCollection / extract single trajectory

        if not isinstance(mode, list):
            modes = [mode]
        else:
            modes = mode

        if len(modes) == 0:
            raise ValueError("need at least one mode")

        wrong_modes = [mode for mode in modes if mode not in decoders]
        if wrong_modes:
            raise ValueError(
                f"unknown {'mode' if len(modes) == 1 else 'modes'}: "
                + (mode if len(modes) == 1 else ", ".join(repr(mode) for mode in modes))
                + "."
                + " Choose one of {{{', '.join(sorted(decoders))}}}."
            )

        def maybe_show_progress(modes):
            if not progress:
                return modes

            return utils.progress_status(modes)

        decoded = [
            pipe(
                [X, states],
                decoders.get(mode),
                lambda x: x.compute(),
                curry(to_trajectory, name=mode),
            )
            for mode in maybe_show_progress(modes)
        ]

        if len(decoded) > 1:
            return mpd.TrajectoryCollection(decoded)

        return decoded[0]
