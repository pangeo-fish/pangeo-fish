import dask
import dask.array as da
import dask_image.ndfilters
import numpy as np
import scipy.ndimage


def gaussian_filter(X, sigma, **kwargs):
    if isinstance(X, da.Array) and X.npartitions > 1:
        return dask_image.ndfilters.gaussian_filter(X, sigma=sigma, **kwargs)
    elif isinstance(X, da.Array):
        return X.map_blocks(
            scipy.ndimage.gaussian_filter,
            sigma=sigma,
            meta=np.array((), dtype=X.dtype),
            **kwargs,
        )
    else:
        return scipy.ndimage.gaussian_filter(X, sigma=sigma, **kwargs)


def predict(X, sigma, *, mask=None, **kwargs):
    filter_kwargs = {"mode": "constant", "cval": 0} | kwargs
    filtered = gaussian_filter(X, sigma=sigma, **filter_kwargs)

    if mask is None:
        return filtered

    return np.where(mask, filtered, 0)


def score(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    truncate=4.0,
):
    """score of a single pass (forwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    sigma : float
        standard deviation of the gaussian kernel, in units of pixels
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
        initial = dask.compute(initial_probability)
    else:
        initial = initial_probability
    if isinstance(mask, da.Array):
        mask = dask.compute(mask)

    normalizations.append(np.sum(initial * dask.compute(emission[0, ...])[0]))
    previous = initial

    for index in range(1, n_max):
        prediction = predict(
            previous,
            sigma=sigma,
            mask=mask,
            truncate=truncate,
        )
        updated = prediction * dask.compute(emission[index, ...])[0]

        normalizations.append(np.sum(updated))
        normalized = updated / normalizations[index]

        previous = normalized

    normalizations_ = np.stack(normalizations, axis=0)

    return -np.sum(np.log(normalizations_))


def forward(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    truncate=4.0,
):
    """single pass (forwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    sigma : float
        standard deviation of the gaussian kernel, in units of pixels
    initial_probability : array-like
        The probability of the first hidden state
    mask : array-like, optional
        A mask to apply after each step. No shadowing yet.

    Returns
    -------
    score : float
        A measure of how well the model parameter fits the data.
    """
    n_max = emission.shape[0]

    predictions = []
    states = []

    predictions.append(initial_probability)
    states.append(initial_probability)

    for index in range(1, n_max):
        prediction = predict(
            states[index - 1],
            sigma=sigma,
            mask=mask,
            truncate=truncate,
        )
        predictions.append(prediction)

        updated = prediction * emission[index, ...]

        normalized = updated / np.sum(updated)
        states.append(normalized)

    return np.stack(predictions, axis=0), np.stack(states, axis=0)


def backward(
    states,
    predictions,
    sigma,
    mask=None,
    *,
    truncate=4.0,
):
    n_max = states.shape[0]
    eps = 2.204e-16**20

    smoothed = [states[-1, ...]]
    backward_predictions = [states[-1, ...]]
    for index in range(1, n_max):
        ratio = smoothed[index - 1] / (predictions[-index, ...] + eps)
        backward_prediction = predict(ratio, sigma=sigma, mask=None, truncate=truncate)
        normalized = backward_prediction / np.sum(backward_prediction)
        backward_predictions.append(normalized)

        updated = normalized * states[-index - 1, ...]
        updated_normalized = updated / np.sum(updated)

        smoothed.append(updated_normalized)

    return np.stack(backward_predictions[::-1], axis=0), np.stack(
        smoothed[::-1], axis=0
    )


def forward_backward(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    truncate=4.0,
):
    """double pass (forwards and backwards) of the spatial HMM filter

    Parameters
    ----------
    emission : array-like
        probability density function of the observations (emission probabilities)
    sigma : float
        standard deviation of the gaussian kernel, in units of pixels
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
        sigma=sigma,
        initial_probability=initial_probability,
        mask=mask,
        truncate=truncate,
    )
    backwards_predictions, backwards_states = backward(
        states=forward_states,
        predictions=forward_predictions,
        sigma=sigma,
        mask=mask,
        truncate=truncate,
    )
    return backwards_states
