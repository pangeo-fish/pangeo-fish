import warnings

import dask
import dask.array as da
import numpy as np
import zarr  # noqa: F401
from tqdm import tqdm


def score(emission, predictor, initial_probability, mask=None):
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

    initial = initial_probability

    normalizations.append(np.sum(initial * emission[0, ...]))
    previous = initial

    for index in tqdm(range(1, n_max), desc="Scoring"):
        prediction = predictor.predict(previous)
        updated = prediction * emission[index, ...]

        normalization_factor = np.sum(updated)

        if normalization_factor == 0:
            warnings.warn(
                f"Empty product of the prediction with the true distribution at step {index+1}.",
                RuntimeWarning,
            )
            return 1e6
        normalizations.append(normalization_factor)
        normalized = updated / normalizations[index]

        previous = normalized

    normalizations_ = np.stack(normalizations, axis=0)

    return -np.sum(np.log(normalizations_))


import warnings

import dask.array as da
import numpy as np
from tqdm import tqdm


def score_final_pos(
    emission,
    predictor,
    initial_probability,
    final_probability,
    mask,
    *,
    return_states=False,
    eps=1e-12,
):
    """
    Forward pass with diagnostics + several policies to handle updated.sum()==0.
    Returns loss (or loss, preds_stack, states_stack if return_states True).
    """
    n_max = emission.shape[0]
    predictions = [initial_probability]
    states = [initial_probability]

    normalizations = []

    obs0 = emission[0, ...]
    if isinstance(obs0, da.Array):
        obs0 = obs0.compute()
    if obs0.ndim > 1:
        obs0 = obs0[0]
    norm0 = float(np.sum(initial_probability * obs0))
    normalizations.append(norm0 if norm0 > 0 else eps)

    final = emission[-1, ...]
    final_idx_max = int(np.argmax(final))
    final_val_max = float(np.max(final))
    print(
        f"[score_final_pos] final_probability max index = {final_idx_max}, value = {final_val_max}"
    )

    for index in tqdm(range(1, n_max), desc="Forward pass"):
        # prediction
        prediction = predictor.predict(states[index - 1], mask=mask)
        if isinstance(prediction, da.Array):
            prediction = prediction.compute()

        prediction /= np.sum(prediction) + 1e-16

        predictions.append(prediction)

        # emission
        obs = emission[index, ...]
        if isinstance(obs, da.Array):
            obs = obs.compute()
        if obs.ndim > 1:
            obs = obs[0]

        updated = prediction * obs
        norm_factor = float(np.sum(updated))

        if norm_factor == 0:
            normalizations.append(eps)
            print(f"[WARNING] Step {index}: sum(updated)==0 -> keeping previous state")

        else:
            normalizations.append(norm_factor)
            normalized = updated / (norm_factor + 1e-16)

        states.append(normalized)

    preds_stack = np.stack(predictions, axis=0)
    states_stack = np.stack(states, axis=0)

    last_state = states_stack[-1, ...]
    print(
        f"[score_final_pos] last_state shape = {last_state.shape}, sum={float(np.sum(last_state)):.3e}, max={float(np.max(last_state)):.3e}"
    )

    state_val = float(states_stack[-1, final_idx_max])
    if state_val <= 0:
        warnings.warn(
            f"state_val at index {final_idx_max} is {state_val} -> using eps for loss",
            RuntimeWarning,
        )

    loss = -np.log(state_val + eps)

    final_state_val = float(states_stack[-1, final_idx_max])
    final_info = {
        "final_index": final_idx_max,
        "final_value": final_val_max,
        "state_value": final_state_val,
    }
    print(final_info)
    return loss


# def score_final_pos(emission, predictor, initial_probability, final_probability, mask,
#                     return_states=False, debug=True, debug_interval=50):
#     """
#     Version minimale avec tests rapides d'overlap insérés dans la boucle.
#     - debug_interval : fréquence d'affichage (tous les N pas)
#     - debug=True pour activer les diagnostics
#     """
#     n_max = emission.shape[0]
#     predictions = [initial_probability]
#     states = [initial_probability]

#     final = final_probability
#     final_idx_max = int(np.argmax(final))
#     final_val_max = float(np.max(final))
#     if debug:
#         print(f"[score_final_pos] final_probability max index = {final_idx_max}, value = {final_val_max}")

#     for index in tqdm(range(1, n_max), desc="Forward pass"):
#         # prediction
#         prediction = predictor.predict(states[index - 1], mask=mask)
#         # if prediction is dask array, compute it
#         if isinstance(prediction, da.Array):
#             prediction = prediction.compute()

#         predictions.append(prediction)

#         # emission (observation) pour ce pas
#         obs = emission[index, ...]
#         if isinstance(obs, da.Array):
#             obs = obs.compute()
#         if obs.ndim > 1:
#             obs = obs[0]  # forcer shape correcte si une dim en trop persiste

#         # update
#         updated = prediction * obs
#         norm_factor = np.sum(updated)

#         # --- TEST RAPIDE d'OVERLAP (executé périodiquement, au dernier pas, ou si norm_factor == 0)
#         if debug and (index % debug_interval == 0 or index == n_max - 1 or norm_factor == 0):
#             try:
#                 # counts non-zero (plus rapide que flatnonzero pour juste compter)
#                 pred_nz_count = int(np.count_nonzero(prediction > 0))
#                 obs_nz_count = int(np.count_nonzero(obs > 0))
#                 print(f"[DEBUG_OVERLAP] Step {index}: pred_nz={pred_nz_count}, obs_nz={obs_nz_count}")

#                 # argmaxs
#                 argmax_pred = int(np.argmax(prediction))
#                 argmax_obs = int(np.argmax(obs))
#                 print(f"[DEBUG_OVERLAP] argmax_pred={argmax_pred}, argmax_obs={argmax_obs}")

#                 # valeur de prediction au niveau de l'argmax de l'émission
#                 pred_at_obs_argmax = float(prediction[argmax_obs])
#                 print(f"[DEBUG_OVERLAP] prediction at emission argmax = {pred_at_obs_argmax:.3e}")

#                 # intersection only if arrays of nz indices are not too large
#                 # (otherwise np.intersect1d would être très coûteux)
#                 # compute flatnonzero only if sizes raisonnables
#                 MAX_PAIRWISE = 2e8  # seuil produit sizes pour autoriser l'intersection
#                 if pred_nz_count > 0 and obs_nz_count > 0 and (pred_nz_count * obs_nz_count) <= MAX_PAIRWISE and pred_nz_count < 200000 and obs_nz_count < 200000:
#                     pred_nz_idx = np.flatnonzero(prediction > 0)
#                     obs_nz_idx = np.flatnonzero(obs > 0)
#                     inter = np.intersect1d(pred_nz_idx, obs_nz_idx)
#                     print(f"[DEBUG_OVERLAP] intersection size = {inter.size}")
#                 else:
#                     if pred_nz_count == 0 or obs_nz_count == 0:
#                         print(f"[DEBUG_OVERLAP] no non-zero in one of arrays (skip intersection)")
#                     else:
#                         print(f"[DEBUG_OVERLAP] intersection test skipped (too large: {pred_nz_count} * {obs_nz_count})")
#             except Exception as e:
#                 print(f"[DEBUG_OVERLAP] exception during overlap test: {e}")

#         # log updated stats occasionally
#         if debug and (index % debug_interval == 0 or index == n_max - 1):
#             print(f"[DEBUG] Step {index}: prediction sum={np.sum(prediction):.3e}, max={np.max(prediction):.3e}, min={np.min(prediction):.3e}")
#             print(f"[DEBUG] Step {index}: emission sum={np.sum(obs):.3e}, max={np.max(obs):.3e}, min={np.min(obs):.3e}")
#             print(f"[DEBUG] Step {index}: updated sum={norm_factor:.3e}, max={np.max(updated):.3e}, min={np.min(updated):.3e}")

#         # warning / fallback
#         if norm_factor == 0:
#             print(f"[WARNING] Step {index}: sum(updated) = 0 (sum(prediction)={np.sum(prediction):.3e}, sum(emission)={np.sum(obs):.3e})")
#             # fallback minimal conservateur : garder l'état précédent pour éviter extinction complète
#             normalized = states[-1]
#         else:
#             normalized = updated / (norm_factor + 1e-16)

#         states.append(normalized)

#     preds_stack = np.stack(predictions, axis=0)
#     states_stack = np.stack(states, axis=0)

#     # dernier état pour info
#     last_state = states_stack[-1, ...]
#     if debug:
#         print(f"[score_final_pos] last_state shape = {last_state.shape}, sum={float(np.sum(last_state)):.3e}, max={float(np.max(last_state)):.3e}")

#     # calcul du loss basé sur final_probability index (alignement par index supposé)
#     state_val = states_stack[-1, final_idx_max]
#     if state_val <= 0:
#         warnings.warn(f"state_val at index {final_idx_max} is {state_val} -> returning large loss", RuntimeWarning)
#     loss = -np.log(state_val + 1e-16)

#     if return_states:
#         return loss, preds_stack, states_stack
#     return loss


def forward(emission, predictor, initial_probability, mask):
    n_max = emission.shape[0]
    predictions = [initial_probability]
    states = [initial_probability]

    for index in tqdm(range(1, n_max), desc="Forward pass"):
        prediction = predictor.predict(states[index - 1], mask=mask)
        predictions.append(prediction)
        updated = prediction * emission[index, ...]
        norm_factor = np.sum(updated)
        if norm_factor == 0:
            print(f"Step {index}: sum(updated) = 0")
        normalized = updated / (norm_factor + 1e-16)  # évite NaN
        states.append(normalized)

    return np.stack(predictions, axis=0), np.stack(states, axis=0)


def backward(states, predictions, predictor, mask):
    n_max = states.shape[0]
    eps = 2.204e-16**20

    smoothed = [states[-1, ...]]
    backward_predictions = [states[-1, ...]]

    for index in tqdm(range(1, n_max), desc="Backward pass"):
        ratio = smoothed[index - 1] / (predictions[-index, ...] + eps)
        backward_prediction = predictor.predict(ratio, mask=mask)
        normalized = backward_prediction / np.sum(backward_prediction)
        backward_predictions.append(normalized)

        updated = normalized * states[-index - 1, ...]
        updated_normalized = updated / np.sum(updated)
        smoothed.append(updated_normalized)

    return (
        np.stack(backward_predictions[::-1], axis=0),
        np.stack(smoothed[::-1], axis=0),
    )


def forward_backward(emission, predictor, initial_probability, mask=None):
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
    score : float
        A measure of how well the model parameter fits the data.
    """

    forward_predictions, forward_states = forward(
        emission=emission,
        predictor=predictor,
        initial_probability=initial_probability,
        mask=mask,
    )

    backwards_predictions, backwards_states = backward(
        states=forward_states,
        predictions=forward_predictions,
        predictor=predictor,
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
