import warnings

import numpy as np


def rgb_to_cmyk(colors):
    # Ensure format and dimensions
    colors = np.array(colors)
    single_color = False
    if len(colors.shape) == 1:
        colors = np.expand_dims(colors, 0)
        single_color = True
    colors = colors[:, :3]

    # Determine black key
    k = 1 - colors.max(1)

    # Colors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = (1 - colors[:, 0] - k) / (1 - k)
        m = (1 - colors[:, 1] - k) / (1 - k)
        y = (1 - colors[:, 2] - k) / (1 - k)

    # Fix nans (where black)
    c[np.isnan(c)] = 0
    m[np.isnan(m)] = 0
    y[np.isnan(y)] = 0

    # Collect
    new_colors = np.stack([c, m, y, k], 1)

    # If only one color was passed
    if single_color:
        new_colors = np.squeeze(new_colors, 0)

    return new_colors


def cmyk_to_rgb(colors):
    # Ensure format and dimensions
    colors = np.array(colors)
    single_color = False
    if len(colors.shape) == 1:
        colors = np.expand_dims(colors, 0)
        single_color = True

    r = (1 - colors[:, 0]) * (1 - colors[:, 3])
    g = (1 - colors[:, 1]) * (1 - colors[:, 3])
    b = (1 - colors[:, 2]) * (1 - colors[:, 3])

    # Collect
    new_colors = np.stack([r, g, b], 1)

    # If only one color was passed
    if single_color:
        new_colors = np.squeeze(new_colors, 0)

    return new_colors