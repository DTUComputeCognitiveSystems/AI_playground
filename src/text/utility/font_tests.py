from matplotlib import pyplot as plt, ticker as plticker

from src.text.utility.text_plots import font_size2height, text2cumul_width
from src.utility.matplotlib_utility import points2ax_units


def test_string_plot_length(grid_interval=1.0, fontsize=15, fontname="serif", s='abracadabra`|', text_pos=(1, 7)):
    """
    Tests how the computed height and width of a string is.
    Note: The ` and |  character creates the whole height of the serif font.
    :param grid_interval:
    :param fontsize:
    :param fontname:
    :param s:
    :param text_pos:
    :return:
    """

    # Make a figure
    plt.close("all")
    fig = plt.figure()
    ax = plt.gca()
    renderer = fig.canvas.get_renderer()

    # Set limits
    plt.xlim(0, 14)
    plt.ylim(0, 14)

    # Set grid
    loc = plticker.MultipleLocator(base=grid_interval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='major', axis='both', linestyle='-')

    # Make some text
    text = plt.text(*text_pos, s, fontsize=fontsize, fontname=fontname)
    fig.canvas.draw()

    # Get bounding box
    bb = text.get_window_extent(renderer=renderer)

    # Get span
    x_min = bb.corners()[:, 0].min()
    x_max = bb.corners()[:, 0].max()
    y_min = bb.corners()[:, 1].min()
    y_max = bb.corners()[:, 1].max()
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Compute bounding box height and width
    bb_width, bb_height = points2ax_units([x_span, y_span]).flatten()

    # Compute height of font
    computed_height = font_size2height(fontsize=fontsize, fontname=fontname)

    # Compute cumulative widths
    cumulative_widths = text2cumul_width(s, fontname=fontname, fontsize=fontsize)

    # Get width
    computed_width = cumulative_widths[-1]

    # Print results
    print("{:.2f}, {:.2f}, {:.2f} | {:.2f}, {:.2f}, {:.2f}".format(
        bb_width, computed_width, abs(bb_width - computed_width) / bb_width,
        bb_height, computed_height, abs(bb_height - computed_height) / bb_height,
    ))