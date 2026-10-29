import warnings

import dask
import dask.array as da
import numpy as np
import zarr  # noqa: F401


def score(emission, predictors, predictor_indices, initial_probability, mask=None):
    """Score of a single pass (forwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    predictor: Predictor
        Algorithm for predicting the next time step.
    initial_probability : array-like
        The probability of the first hidden state
    final_probability : array-like, optional
        The probability of the last hidden state
    mask : array-like, optional
        A mask to apply after each step. No shadowing yet.

    Returns
    -------
    score : float
        A measure of how well the model parameter fits the data.
    """
    n_max = emission.shape[0]

    normalizations = []

    initial, mask = dask.compute(initial_probability, mask)
    if isinstance(initial_probability, da.Array):
        [initial] = dask.compute(initial_probability)
    else:
        initial = initial_probability
    if isinstance(mask, da.Array):
        mask = dask.compute(mask)

    normalizations.append(np.sum(initial * dask.compute(emission[0, ...])[0]))
    previous = initial

    for time_index, predictor_index in zip(range(1, n_max), predictor_indices):
        prediction = predictors[predictor_index].predict(previous, mask=mask)
        updated = prediction * dask.compute(emission[time_index, ...])[0]

        normalization_factor = np.sum(updated)
        if normalization_factor == 0:
            warnings.warn(
                f"Empty product of the prediction with the true distribution at step {time_index+1}.",
                RuntimeWarning,
            )
            return 1e6
        normalizations.append(normalization_factor)
        normalized = updated / normalizations[time_index]

        previous = normalized

    normalizations_ = np.stack(normalizations, axis=0)

    return -np.sum(np.log(normalizations_))


def score_final_pos(
    emission,
    predictors,
    predictor_indices,
    initial_probability,
    mask,
    *,
    eps=np.finfo(np.float64).tiny,
):
    """Score of a single pass (forwards) of the spatial HMM filter, based on the final position

    Runs the forward filter and returns the negative log of the filtered probability
    of the cell where the last emission is maximal. The lower, the better the model
    reaches the observed final position.

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    predictors : sequence of Predictor
        Algorithms for predicting the next time step.
    predictor_indices : array-like
        Predictor to use for each transition.
    initial_probability : array-like
        The probability of the first hidden state
    mask : array-like
        A mask to apply after each step. No shadowing yet.
    eps : float, default: np.finfo(np.float64).tiny
        Lower bound used to avoid ``log(0)``.

    Returns
    -------
    score : float
        Negative log of the filtered probability at the most likely final cell
        (``-log(states[-1][argmax(emission[-1])])``). Lower is better.
    """
    n_max = emission.shape[0]

    obs0 = emission[0, ...]
    if hasattr(obs0, "compute"):
        obs0 = obs0.compute()

    updated0 = initial_probability * obs0
    norm0 = float(np.sum(updated0))
    log_total = np.log(norm0) if norm0 > 0 else np.log(eps)
    previous = (
        updated0 / norm0 if norm0 > 0 else initial_probability
    )  # RENORMALISÉ, reste dans une plage saine

    for index, predictor_index in zip(range(1, n_max), predictor_indices):
        prediction = predictors[predictor_index].predict(previous, mask=mask)
        if hasattr(prediction, "compute"):
            prediction = prediction.compute()

        obs = emission[index, ...]
        if hasattr(obs, "compute"):
            obs = obs.compute()

        updated = prediction * obs
        norm_factor = float(np.sum(updated))

        if norm_factor > 0:
            log_total += np.log(norm_factor)
            previous = (
                updated / norm_factor
            )  # renormalisé à chaque pas -> jamais de sous-flottant
        else:
            log_total += np.log(eps)

    final_idx_max = int(np.argmax(emission[-1, ...]))
    state_val = float(
        previous[final_idx_max]
    )  # previous est ici states[-1] NORMALISÉ (somme=1)

    loss = -np.log(max(state_val, eps))  # -log_total

    return loss


def forward(emission, predictors, predictor_indices, initial_probability, mask=None):
    """Single pass (forwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    predictor: Predictor
        Algorithm for predicting the next time step.
    initial_probability : array-like
        The probability of the first hidden state
    mask : array-like, optional
        A mask to apply after each step. No shadowing yet.

    Returns
    -------
    forward_predictions : array-like
        Forward predictions
    smoothed : array-like
        Smoothed states
    """
    n_max = emission.shape[0]

    predictions = []
    states = []

    predictions.append(initial_probability)
    states.append(initial_probability)

    for time_index, predictor_index in zip(range(1, n_max), predictor_indices):
        prediction = predictors[predictor_index].predict(
            states[time_index - 1], mask=mask
        )
        predictions.append(prediction)

        updated = prediction * emission[time_index, ...]

        normalized = updated / np.sum(updated)
        states.append(normalized)

    return np.stack(predictions, axis=0), np.stack(states, axis=0)


def backward(states, predictions, predictors, predictor_indices, mask=None):
    """Single pass (backwards) of the spatial HMM smoother

    Parameters
    ----------
    states : array-like
        Filtered states from the forward pass
    predictions : array-like
        Predictions from the forward pass
    predictors : sequence of Predictor
        Algorithms for predicting the previous time step
    predictor_indices : array-like
        Predictor to use for each transition, in forward order (reversed internally)
    mask : array-like, optional
        Currently unused

    Returns
    -------
    backward_predictions : array-like
        Backward predictions
    smoothed : array-like
        Smoothed states
    """
    n_max = states.shape[0]
    eps = 2.204e-16**20

    smoothed = [states[-1, ...]]
    backward_predictions = [states[-1, ...]]
    for time_index, predictor_index in zip(
        range(1, n_max), predictor_indices[: n_max - 1][::-1]
    ):
        ratio = smoothed[time_index - 1] / (predictions[-time_index, ...] + eps)
        backward_prediction = predictors[predictor_index].predict(ratio, mask=None)
        normalized = backward_prediction / np.sum(backward_prediction)
        backward_predictions.append(normalized)

        updated = normalized * states[-time_index - 1, ...]
        updated_normalized = updated / np.sum(updated)

        smoothed.append(updated_normalized)

    return np.stack(backward_predictions[::-1], axis=0), np.stack(
        smoothed[::-1], axis=0
    )


def forward_backward(
    emission, predictors, predictor_indices, initial_probability, mask=None
):
    """Double pass (forwards and backwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    predictor: Predictor
        Algorithm for predicting the next time step.
    initial_probability : array-like
        The probability of the first hidden state
    final_probability : array-like, optional
        The probability of the last hidden state
    mask : array-like, optional
        A mask to apply after each step. No shadowing yet.

    Returns
    -------
    predictions : array-like
        Forward_Backward predictions
    """

    forward_predictions, forward_states = forward(
        emission=emission,
        predictors=predictors,
        predictor_indices=predictor_indices,
        initial_probability=initial_probability,
        mask=mask,
    )
    _, backwards_states = backward(
        states=forward_states,
        predictions=forward_predictions,
        predictors=predictors,
        predictor_indices=predictor_indices,
        mask=mask,
    )
    return backwards_states


def copy_coords_of(arr, source, dest):
    for name in arr.attrs["coordinates"].split():
        dest[name] = source[name]
        dest[name].attrs.update(source[name].attrs)


def empty_like(group, name, arr, **kwargs):
    new = group.empty_like(name, arr, **kwargs)
    new.attrs.update(arr.attrs)

    return new


def track(sequence, *, display=False, **kwargs):
    if not display:
        return sequence

    try:
        from rich.progress import track as rich_track
    except ImportError as e:
        if display:
            raise ValueError("cannot display the progress bar") from e

    return rich_track(sequence, **kwargs)


def _forward_zarr(ingroup, outgroup, predictor, progress=False):
    """Single pass (forwards) of the spatial HMM filter while writing to zarr

    Parameters
    ----------
    ingroup: zarr.Group
        zarr group containing:

        - the probability density function of the observations (emission probabilities)
        - the initial probability
        - (optionally) a mask to apply after each step. No shadowing yet.

    outgroup : zarr.Group
        Zarr object to write the result to.
    predictor : Predictor
        Algorithm for predicting the next time step.
    progress : bool, default: False
        Whether to display a progress bar.
    """

    emission = ingroup["pdf"]
    initial_probability = ingroup["initial"]
    mask = ingroup.get("mask")

    copy_coords_of(emission, ingroup, outgroup)

    predictions = empty_like(outgroup, "predictions", emission)
    states = empty_like(outgroup, "states", emission)
    normalizations = outgroup.empty(
        "normalizations", dtype=emission.dtype, shape=emission.shape[:1]
    )
    normalizations.attrs["_ARRAY_DIMENSIONS"] = emission.attrs["_ARRAY_DIMENSIONS"][:1]

    predictions[0, ...] = initial_probability
    states[0, ...] = initial_probability
    normalizations[0] = 1  # this is constant, so doesn't make a difference

    n_max = emission.shape[0]
    for index in track(
        range(1, n_max), description="forwards filter", display=progress
    ):
        prediction = predictor.predict(states[index - 1, ...], mask=mask)
        predictions[index, ...] = prediction

        updated = prediction * emission[index, ...]
        normalizations[index] = np.sum(updated)
        normalized = updated / normalizations[index]
        states[index, ...] = normalized

    return outgroup


def _backward_zarr(ingroup, outgroup, predictor, progress=False):
    """Single pass (backwards) of the spatial HMM filter while writing to zarr

    Parameters
    ----------
    ingroup: zarr.Group
        zarr group containing:

        - the probability density function of the observations (emission probabilities)
        - the initial probability
        - (optionally) a mask to apply after each step. No shadowing yet.

    outgroup : zarr.Group
        Zarr object to write the result to.
    predictor: Predictor
        Algorithm for predicting the next time step.
    progress : bool, default: False
        Whether to display a progress bar.
    """
    eps = 2.204e-16**20

    predictions = ingroup["predictions"]
    states = ingroup["states"]

    copy_coords_of(states, ingroup, outgroup)

    smoothed = empty_like(outgroup, "states", states)
    backward_pred = empty_like(outgroup, "predictions", states)

    smoothed[-1, ...] = states[-1, ...]
    backward_pred[-1, ...] = states[-1, ...]

    n_max = states.shape[0]
    for index in track(
        range(1, n_max), description="backwards filter", display=progress
    ):
        ratio = smoothed[-index, ...] / (predictions[-index, ...] + eps)
        backward_prediction = predictor.predict(ratio, mask=None)
        normalized = backward_prediction / np.sum(backward_prediction)
        backward_pred[-index - 1, ...] = normalized

        updated = normalized * states[-index - 1, ...]
        updated_normalized = updated / np.sum(updated)

        smoothed[-index - 1, ...] = updated_normalized

    return outgroup
