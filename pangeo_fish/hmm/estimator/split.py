from dataclasses import asdict, dataclass, replace

import numpy as np
import xarray as xr
from tlz.functoolz import compose_left, curry
from tlz.itertoolz import identity

from ... import utils
from ...tracks import to_trajectory
from ..decode import mean_track, modal_track, viterbi
from ..filter import forward, score


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
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            return score(
                emission,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        value = xr.apply_ufunc(
            _algorithm,
            X.pdf.fillna(0),
            X.mask,
            X.initial,
            final,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[()],
            dask="allowed",
        )
        return value.fillna(np.inf)

    def _forward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            return forward(
                emission=emission,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            final,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[temporal_dims + spatial_dims],
            dask="allowed",
        )

    def _forward_backward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            forward_state = forward(
                emission=emission,
                sigma=sigma,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                truncate=truncate,
            )
            backward_state = forward(
                emission=forward_state[::-1, ...],
                sigma=sigma,
                mask=mask,
                initial_probability=forward_state[-1, ...],
                final_probability=forward_state[0, ...],
                truncate=truncate,
            )

            return backward_state[::-1, ...]

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            final,
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
            - `final`, the final probability map
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
            - `final`, the final probability map (optional)
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
        *,
        mode="viterbi",
        spatial_dims=None,
        temporal_dims=None,
        is_states=False,
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
        mode : {"mean", "mode", "viterbi"}, default: "viterbi"
            The decoding method. Can be one of
            - ``"mean"``: use the centroid of the state probabilities as decoded state
            - ``"mode"``: use the maximum of the state probabilities as decoded state
            - ``"viterbi"``: use the viterbi algorithm to determine the most probable states
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.
        is_states : bool, default: False
            Whether ``X`` is the precomputed state probabilities. Useful if computing both
            the mean and mode states.
        """

        if is_states and mode == "viterbi":
            raise ValueError("cannot pass state probabilities to the viterbi algorithm")
        elif is_states and mode != "viterbi":
            compute_states = curry(
                self.predict_proba,
                spatial_dims=spatial_dims,
                temporal_dims=temporal_dims,
            )
        else:
            compute_states = identity

        decoders = {
            "mean": compose_left(compute_states, mean_track),
            "mode": compose_left(compute_states, modal_track),
            "viterbi": curry(viterbi, sigma=self.sigma),
        }

        decoder = decoders.get(mode)
        if decoder is None:
            raise ValueError(
                f"unknown mode: {mode!r}. Choose one of {{{', '.join(sorted(decoders))}}}"
            )

        decoded = decoder(X)

        return to_trajectory(decoded)
