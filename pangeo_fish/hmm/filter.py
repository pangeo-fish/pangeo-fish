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


def single_pass(
    pdf, sigma, initial_probability, mask=None, *, final_probability=None, truncate=4.0
):
    """compute a single pass (forwards) of the HMM filter

    Parameters
    ----------
    pdf : array-like
        probability density function of the observations
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
    predictions : array-like
        The probability of the
    normalizations : array-like
        The normalization factors per time step.
    posterior_probabilities : array-like
        The probability of the hidden state given the observation.
    """
    n_max = pdf.shape[0] - (0 if final_probability is None else 1)

    normalizations = []
    posterior_probabilities = []

    posterior_probabilities.append(initial_probability)
    normalizations.append(np.sum(initial_probability * pdf[0, ...]))

    for index in range(1, n_max):
        prediction = predict(
            posterior_probabilities[index - 1],
            sigma=sigma,
            mask=mask,
            truncate=truncate,
        )
        updated = prediction * pdf[index, ...]

        normalizations.append(np.sum(updated))
        normalized = updated / normalizations[index]

        posterior_probabilities.append(normalized)

    if final_probability is not None:
        normalizations.append(np.sum(final_probability))
        posterior_probabilities.append(final_probability)

    normalizations_ = np.stack(normalizations, axis=0)
    posterior_probabilities_ = np.stack(posterior_probabilities, axis=0)

    return normalizations_, posterior_probabilities_


def score(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    final_probability=None,
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
    n_max = emission.shape[0] - (0 if final_probability is None else 1)

    normalizations = []

    initial, final, mask = dask.compute(initial_probability, final_probability, mask)

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

    if final is not None:
        normalizations.append(np.sum(final))

    normalizations_ = np.stack(normalizations, axis=0)

    return -np.sum(np.log(normalizations_))


def forward(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    final_probability=None,
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
    final_probability : array-like, optional
        The probability of the last hidden state
    mask : array-like, optional
        A mask to apply after each step. No shadowing yet.

    Returns
    -------
    score : float
        A measure of how well the model parameter fits the data.
    """
    n_max = emission.shape[0] - (0 if final_probability is None else 1)

    state = []

    state.append(initial_probability)

    for index in range(1, n_max):
        prediction = predict(
            state[index - 1],
            sigma=sigma,
            mask=mask,
            truncate=truncate,
        )
        updated = prediction * emission[index, ...]

        normalized = updated / np.sum(updated)

        state.append(normalized)

    if final_probability is not None:
        state.append(final_probability)

    return np.stack(state, axis=0)


def forward_backward(
    emission,
    sigma,
    initial_probability,
    mask=None,
    *,
    final_probability=None,
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

    forward_state = forward(
        emission=emission,
        sigma=sigma,
        initial_probability=initial_probability,
        mask=mask,
        final_probability=final_probability,
        truncate=truncate,
    )
    backwards_state = forward(
        emission=forward_state[::-1, ...],
        sigma=sigma,
        initial_probability=final_probability,
        mask=mask,
        final_probability=initial_probability,
        truncate=truncate,
    )
    return backwards_state
