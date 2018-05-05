import numpy as np
from matplotlib import pyplot as plt

from src.text.utility.font_specs import Font
from src.utility.matplotlib_utility import ax_aspect_ratio


def height2font_size(y_range, fontname, ax=None):
    """
    Takes a range on the y-scale of a set of axes, with a font name, and computes the needed fontsize for making
    the text exactly that height.
    :param y_range:
    :param fontname:
    :param ax
    :rtype float
    """
    # Get height-fontsize-ratio for font name
    font = Font.get_font(fontname=fontname)

    # Get current axis
    ax = plt.gca() if ax is None else ax

    # Get box-corners of axes in figure and y-corners
    corners = ax.get_window_extent().get_points()
    y_bottom, y_top = corners[:, 1]

    # Get limits of y-axis
    y_min, y_max = ax.get_ylim()

    # Get unit scale
    y_scale = (y_top - y_bottom) / (y_max - y_min)

    # Make font-size
    fontsize = y_range * y_scale / font.hf_ratio

    return fontsize


def font_size2height(fontsize, fontname, ax=None):
    """
    Takes a fontsize and fontname and determines how high the text of that font will be.
    :param fontsize
    :param fontname:
    :param ax
    :rtype float
    """
    # Get height-fontsize-ratio for font name
    font = Font.get_font(fontname=fontname)

    # Get current axis
    ax = plt.gca() if ax is None else ax

    # Get box-corners of axes in figure and y-corners
    corners = ax.get_window_extent().get_points()
    y_bottom, y_top = corners[:, 1]

    # Get scale of y-axis
    y_min, y_max = ax.get_ylim()

    # Get unit scale
    y_scale = (y_top - y_bottom) / (y_max - y_min)

    # Determine height
    y_range = fontsize * font.hf_ratio / y_scale

    return y_range


def mean_char_width(fontname, fontsize, ax=None):
    """
    Returns the with of the mean character in the current axes.
    :param str fontname: Name of font.
    :param float fontsize: Initial font-size.
    :param ax: The axis for plotting. Defaults to plt.gca().
    :return: float
    """
    # Get font
    font = Font.get_font(fontname=fontname)

    # Get height of text
    height = font_size2height(fontsize=fontsize, fontname=fontname, ax=ax)

    # Get aspect
    aspect = ax_aspect_ratio(ax=ax)

    # Determine widths
    width = aspect * height / font.hw_ratio_mean

    return width


def text2cumul_width(text, fontname, fontsize, ax=None):
    """
    Computes the cumulative width of the characters.
    :param str text: Text.
    :param str fontname: Name of font.
    :param float fontsize: Initial font-size.
    :param ax: The axis for plotting. Defaults to plt.gca().
    :return: np.ndarray
    """
    # Get font
    font = Font.get_font(fontname=fontname)

    # Get height of text
    height = font_size2height(fontsize=fontsize, fontname=fontname, ax=ax)

    # Get characters height-width ratios
    ratios = [font.hw_ratios.get(val, font.hw_ratio_mean) for val in text]

    # Get aspect
    aspect = ax_aspect_ratio(ax=ax)

    # Determine widths
    widths = [aspect * height / ratio for ratio in ratios]

    # Determine cumulative widths
    cumulative_widths = np.cumsum(widths)

    return cumulative_widths