import textwrap

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


def reduce_font_size(fontsize):
    if fontsize > 30:
        return fontsize - 3
    elif fontsize > 15:
        return fontsize - 2
    elif fontsize > 1:
        return fontsize - 1
    return fontsize * 0.1


def flow_text_into_axes(text, x=None, y=None, fontsize=15, fontname="serif", line_spacing=1.1,
                        right_relative_margin=0.075, ax=None):
    """
    Flows some text into a set of axes.
    Font-size will be reduced if text does not fit.
    :param str text: The text to show.
    :param float x: Left coordinate of text.
    :param float y: Top coordinate of text.
    :param float fontsize: Initial font-size.
    :param str fontname: Name of font.
    :param float line_spacing: Relative spacing between lines.
    :param float right_relative_margin: Relative margin on the left. Used to avoid text-overflow.
    :param ax: The axis for plotting. Defaults to plt.gca().
    """
    # Default axes
    ax = plt.gca() if ax is None else ax

    # Line limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Default x and y
    if x is None:
        x = (x_max - x_min) * 0.1
    if y is None:
        y = y_max - (y_max - y_min) * 0.1

    success = False
    while not success:
        success = True

        # Compute height of text
        height = font_size2height(fontsize=fontsize, fontname=fontname)

        # Compute characters per line
        width = len(text.replace("\n", "")) * mean_char_width(fontname=fontname, fontsize=fontsize, ax=ax)
        width_per_character = width / len(text)
        characters_per_line = int((1 - right_relative_margin) * (x_max - x) / width_per_character)

        # Wrap text
        text_lines = text.split("\n")
        wrapped_text = [
            line
            for lines in text_lines
            for line in
            textwrap.wrap(text=lines, width=characters_per_line, replace_whitespace=False, drop_whitespace=False)
        ]

        # Determine last line end
        last_y = y - len(wrapped_text) * height * line_spacing
        if last_y < y_min:
            success = False
            fontsize = reduce_font_size(fontsize=fontsize)
            continue

        # View lines
        for line_nr, line in enumerate(wrapped_text):
            line = line.strip()

            # Compute y-location
            y = y - line_nr * height * line_spacing

            # View line
            plt.text(x, y, line, fontsize=fontsize, fontname=fontname)


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