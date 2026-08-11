import numpy as np

def clip_bounds(x, bounds):
    """
    Clips the vector x to be within bounds.
    bounds is a tuple (lower_bound, upper_bound)
    """
    lower, upper = bounds
    return np.clip(x, lower, upper)
