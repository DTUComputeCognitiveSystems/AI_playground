import numpy as np
from matplotlib import pyplot as plt


def ax_aspect_ratio(ax=None):
    """
    Returns tha aspect ratio of a set of 2D axes.
    :param ax:
    :return:
    """
    # Get scales
    x_scale, y_scale = ax_points_scale(ax=ax)

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
    # Get scales
    x_scale, y_scale = ax_points_scale(ax)

    # Prepare units
    units = np.array(units)
    units = np.array([units] if len(units.shape) == 1 else units)

    # Compute output
    points = units * np.array([x_scale, y_scale])

    return points


def points2ax_units(points, ax=None):
    """
    Converts from points in figure to the unit of the axes.
    :param points:
    :param ax:
    :return:
    """
    # Get scales
    x_scale, y_scale = ax_points_scale(ax)

    # Prepare points
    points = np.array(points)
    points = np.array([points] if len(points.shape) == 1 else points)

    # Compute output
    units = points / np.array([x_scale, y_scale])

    return units


def ax_points_scale(ax=None):
    """
    Returns the unit-to-points ratio for some axes.
    Thus a distance k on the x-axis is equal to k * x_scale points.
    :param ax:
    :return: float, float
    """
    # Ensure axes
    ax = plt.gca() if ax is None else ax

    # Get axis corners
    corners = ax.get_window_extent().get_points()
    x_bottom, x_top = corners[:, 0]
    y_bottom, y_top = corners[:, 1]

    # Get limits of axes
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    # Get unit scales
    y_scale = (y_top - y_bottom) / (y_max - y_min)
    x_scale = (x_top - x_bottom) / (x_max - x_min)

    return x_scale, y_scale