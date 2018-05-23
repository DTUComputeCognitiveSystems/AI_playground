from matplotlib import pyplot as plt


def ax_aspect_ratio(ax=None):
    """
    Returns tha aspect ratio of a set of 2D axes.
    :param ax:
    :return:
    """
    # Ensure axes
    ax = plt.gca() if ax is None else ax

    # Get limits of axes
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    # Transform into points
    x_min_p, y_min_p = ax_units2points([x_min, y_min])
    x_max_p, y_max_p = ax_units2points([x_max, y_max])

    # Get unit scales
    y_scale = (y_max_p - y_min_p) / (y_max - y_min)
    x_scale = (x_max_p - x_min_p) / (x_max - x_min)

    # Get aspect
    aspect = y_scale / x_scale

    return aspect


def ax_units2points(units, ax=None):
    """
    Converts from the units of some axes to the number of points in figure.
    :param units:
    :param ax:
    :return:
    """
    # Ensure axes
    ax = plt.gca() if ax is None else ax

    # Transform
    return ax.transData.transform(units)


def points2ax_units(points, ax=None):
    """
    Converts from points in figure to the unit of the axes.
    :param points:
    :param ax:
    :return:
    """
    # Ensure axes
    ax = plt.gca() if ax is None else ax

    # Get inverted transform
    inv = ax.transData.inverted()

    # Transform
    return inv.transform(points)
