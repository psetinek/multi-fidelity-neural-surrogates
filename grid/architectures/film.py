"""Dimension-agnostic FiLM modulation.

For an activation tensor h of shape (C, *spatial) and conditioning-
derived vectors scale/shift of shape (C,), apply

    output = (1 + scale) * h + shift

with broadcasting across every spatial axis. The (1 + scale)
parametrization keeps the transform near-identity when the film
projection is near zero at init.
"""


def apply_film(h, scale, shift):
    """Broadcast (scale, shift) ∈ R^C over the spatial axes of h ∈ R^{C, *spatial}.

    Works for any number of spatial dimensions via (C,) + (1,)*ndim
    reshaping.
    """
    ndim = h.ndim - 1
    broadcast_shape = (scale.shape[0],) + (1,) * ndim
    scale = scale.reshape(broadcast_shape)
    shift = shift.reshape(broadcast_shape)
    return (1.0 + scale) * h + shift


def split_film_params(params):
    """Split a (2C,) projection output into (C,) scale + (C,) shift."""
    c = params.shape[0] // 2
    return params[:c], params[c:]
