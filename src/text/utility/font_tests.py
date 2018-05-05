from matplotlib import pyplot as plt, ticker as plticker

from src.text.utility.font_axes import font_size2height, text2cumul_width
from src.text.utility.text_modifiers import TextModifier
from src.text.utility.text_plots import flow_text_into_axes
from src.utility.matplotlib_utility import points2ax_units


def test_string_plot_length(grid_interval=1.0, fontsize=15, fontname="serif",
                            strings=('abracadabra`|', "a string with some words"), text_pos=(1, 7)):
    """
    Tests how the computed height and width of a string is.
    Note: The ` and |  character creates the whole height of the serif font.
    :param grid_interval:
    :param fontsize:
    :param fontname:
    :param text_pos:
    :param strings:
    :return:
    """

    # Header
    print("bb_width, computed_width, diff   | bb_height, computed_height, diff")

    for s in strings:
        # Make a figure
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

        print("{:.4f}  , {:.4f}        , {:.4f} | {:.4f}   , {:.4f}         , {:.4f}".format(
            bb_width, computed_width, abs(bb_width - computed_width) / bb_width,
            bb_height, computed_height, abs(bb_height - computed_height) / bb_height,
        ))


if __name__ == "__main__":

    plt.close("all")

    ###
    # Check dimensions of a string plotted

    do_tests = False

    if do_tests:
        # TODO: Figure out how to handle strings of less-than-full-height and spaces
        test_string_plot_length(
            strings=('abracadabra`|', "a string with some words",
                     "a string with some words`|", "astringwithsomewords`|")
        )

    ###
    # Mark all e's in a text

    # Here's some text from online
    long_text = """
    A flight linking Singapore and the Malaysian capital Kuala Lumpur has become the busiest international route in the world, research shows.
    abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz e e e
    Planes made 30,537 trips between the two airports in the year to February 2018, OAG Aviation said.
    The route overtook Hong Kong-Taipei in a list dominated by Asian destinations.
    Flying between Singapore and Kuala Lumpur takes about an hour, and there are plans to build a high-speed rail link between the two.
    The figures mean an average of 84 flights per day plied the route.
    The route is operated by a host of budget carriers such as Scoot, Jetstar, Air Asia and Malindo Air as well as the two country's flagship carriers Singapore Airlines and Malaysia Airlines.
    """

    # Optionally remove some lines for testing
    long_text = "\n".join(long_text.split("\n")[:])

    # Make figure
    fig = plt.figure()
    ax = plt.gca()
    plt.xlim(0, 14)
    plt.ylim(0, 14)
    ax.set_xticks([])
    ax.set_yticks([])

    # TODO: At the moment the first line can not be empty. Fix that.
    long_text = long_text.strip()

    # Make soem text-modifiers (mark e's blue)
    modifiers = [
        TextModifier(val, val + 1, "color", "blue") for val, char in enumerate(long_text) if char == "e"
    ]

    # Flow text into fiure with markings
    flow_text_into_axes(
        text=long_text,
        modifiers=modifiers,
    )
