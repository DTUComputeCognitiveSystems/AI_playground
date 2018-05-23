import numpy as np
from matplotlib import pyplot as plt

from src.text.utility.font_specs import Font
from src.utility.matplotlib_utility import ax_aspect_ratio, points2ax_units, ax_units2points


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


def text2cumul_width(text, fontname, fontsize, ax=None, mean_is_default=True):
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
    if mean_is_default:
        ratios = [font.hw_ratios.get(val, font.hw_ratio_mean) for val in text]
    else:
        ratios = [font.hw_ratios[val] for val in text]

    # Get aspect
    aspect = ax_aspect_ratio(ax=ax)

    # Determine widths
    widths = [aspect * height / ratio for ratio in ratios]

    # Determine cumulative widths
    cumulative_widths = np.cumsum(widths)

    return cumulative_widths


if __name__ == "__main__":
    plt.close("all")

    fontname = "monospace"

    ###################
    # Text lengths

    print("\n\nText lengths:\n" + "-" * 30)
    fig = plt.figure(figsize=(14, 10))
    renderer = fig.canvas.get_renderer()
    ax = plt.gca()
    ax.set_xlim(0, 1.4)
    ax.set_aspect("equal")

    fontsize = 15

    test_lines = [
        "Suspendisse",
        "Sus p end i sse",
        "Suspendisse potenti.",
        "Suspendisse potenti. Vestibulum vel",
        "Suspendisse potenti. Vestibulum vel turpis libero.",
        "Suspendisse potenti. Vestibulum vel turpis libero. Pellentesque elementum",
        "Suspendisse potenti. Vestibulum vel turpis libero. Pellentesque elementum nisi quam.",
    ]

    for nr, line in enumerate(test_lines):
        y = nr / len(test_lines)

        text = plt.text(0.0, y, line, fontname=fontname, fontsize=fontsize)

        # Get bounding box and determine width and height
        bb = text.get_window_extent(renderer=renderer)

        # Get axes positions
        xlim = plt.xlim()
        ylim = plt.ylim()
        x_pos, y_pos = ax_units2points((xlim[0], ylim[0]), ax=ax)

        estimated_length = text2cumul_width(text=line, fontname=fontname, fontsize=fontsize)
        print("{: .4f}, {:.4f}".format(
            points2ax_units([bb.width + x_pos, bb.height], ax=ax)[0],
            estimated_length[-1])
        )

        plt.axvline(x=estimated_length[-1], ymax=y + 0.025)

    ###################
    # Fontsizes

    fig = plt.figure(figsize=(14, 10))
    renderer = fig.canvas.get_renderer()
    ax = plt.gca()
    ax.set_xlim(0, 1.4)
    ax.set_aspect("equal")

    print("\n\nFontsizes:\n" + "-" * 30)
    line = "Suspendisse potenti. Vestibulum vel turpis libero."
    font_sizes = [5, 8, 10, 12, 15, 18, 24]
    for nr, fontsize in enumerate(font_sizes):
        y = nr / len(font_sizes)

        text = plt.text(0.0, y, line, fontname=fontname, fontsize=fontsize)

        # Get bounding box and determine width and height
        bb = text.get_window_extent(renderer=renderer)

        # Get axes positions
        xlim = plt.xlim()
        ylim = plt.ylim()
        x_pos, y_pos = ax_units2points((xlim[0], ylim[0]), ax=ax)

        estimated_length = text2cumul_width(text=line, fontname=fontname, fontsize=fontsize)
        print("{: .4f}, {:.4f}".format(
            points2ax_units([bb.width + x_pos, bb.height], ax=ax)[0],
            estimated_length[-1])
        )

        plt.axvline(x=estimated_length[-1], ymax=y + 0.025)
