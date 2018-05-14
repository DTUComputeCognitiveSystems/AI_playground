import textwrap
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, transforms

from src.text.utility.font_axes import font_size2height, mean_char_width, text2cumul_width
from src.text.utility.text_modifiers import TextModifier


def plot_modified_line(x, y, text, modifiers, fontsize=None, ax=None, fig=None):
    """
    Plot a line of text with a set of modifiers assign to it.
    :param float x: x-coordinate of text.
    :param float y: y-coordinate of text.
    :param str text: The text.
    :param list[TextModifier] modifiers: The modifiers applied to the text.
    :param ax:
    :param fig:
    """
    # Check for line-break
    if "\n" in text:
        raise ValueError("This method was not made for text with line-breaks.")

    # Get axes and figure
    ax = plt.gca() if ax is None else ax
    fig = plt.gcf() if fig is None else fig

    # Determine all splits for modifiers
    splits = list(sorted(set([
        max(val, 0)
        for modifier in modifiers
        for val in (modifier.start, modifier.end)

    ])))

    # Compute section modifications
    modifier_locs = {val: idx for idx, val in enumerate(splits)}
    section_mods = [dict() for _ in range(len(splits) + 1)]
    for modifier in modifiers:
        # Only consider positive numbers
        start = max(modifier.start, 0)
        end = max(modifier.end, 0)

        # Note modifications for sections.
        for section in section_mods[modifier_locs[start] + 1:modifier_locs[end] + 1]:
            section[modifier.field_name] = modifier.field_value

    # Split text into sections
    sections = [text[start:end] for start, end in zip([0] + splits, splits + [None])]

    # Get axes transform
    t = ax.transData

    # Go through sections
    for section, section_mod in zip(sections, section_mods):
        if fontsize is not None:
            section_mod["fontsize"] = fontsize

        # Write string
        text = ax.text(
            x=x,
            y=y,
            s=section,
            transform=t,
            **section_mod
        )

        # Render text
        text.draw(fig.canvas.get_renderer())

        # Shift transform for next bit of text
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            trans=text._transform,
            x=ex.width,
            units='dots'
        )


def reduce_font_size(fontsize):
    if fontsize > 30:
        return fontsize - 3
    elif fontsize > 15:
        return fontsize - 2
    elif fontsize > 1:
        return fontsize - 1
    return fontsize * 0.1


def break_up_text_for_axes(text, x=None, y=None, fontsize=15, fontname="serif", line_spacing=1.1,
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
        x = (x_max - x_min) * 0.05
    if y is None:
        y = y_max - (y_max - y_min) * 0.1

    # Try with font size and reduce until it fits
    original_newlines = additional_newlines = []
    success = False
    height = font_size2height(fontsize=fontsize, fontname=fontname)
    wrapped_text = [text]
    while not success:
        success = True

        # Compute height of text
        height = font_size2height(fontsize=fontsize, fontname=fontname)

        # # Compute characters per line
        # # TODO: Could be compute exactly as well
        # characters_per_line = int((1 - right_relative_margin) * (x_max - x) /
        #                           mean_char_width(fontname=fontname, fontsize=fontsize, ax=ax))

        # Wrap text
        original_newlines = []
        additional_newlines = []
        wrapped_text = []
        c_length = 0
        # sequences = [val for val in text.split("\n") if val]
        sequences = list(text.split("\n"))
        for sequence_nr, sequence in enumerate(sequences):
            # Note original newline
            if sequence_nr != 0:
                original_newlines.append(c_length)

                # This character counts as well
                c_length += 1

            # Wrap text
            if sequence:
                wrap = wrap_single_line(
                    line=sequence,
                    x=x,
                    fontname=fontname,
                    fontsize=fontsize,
                    relative_right_margin=right_relative_margin,
                    ax=ax
                )
            else:
                wrap = [""]

            # Add new lines
            for nr, line in enumerate(wrap):

                # Note added newline
                if nr != 0:
                    additional_newlines.append(c_length)
                c_length += len(line)

                # Add text
                wrapped_text.append(line)

        # Determine last line end
        last_y = y - len(wrapped_text) * height * line_spacing
        if last_y < y_min:
            success = False
            fontsize = reduce_font_size(fontsize=fontsize)
            continue

    return fontsize, height, wrapped_text, original_newlines, additional_newlines


def wrap_single_line(line, x, fontname, fontsize, relative_right_margin=0.05, ax=None,
                     simple=False):
    # Get horizontal limits
    xmin, xmax = ax.get_xlim()
    x_range = (xmax - x) * (1 - relative_right_margin)

    # Simple and faster method
    if simple:
        characters_per_line = int(x_range / mean_char_width(fontname=fontname, fontsize=fontsize, ax=ax))

        wrap = textwrap.wrap(text=line,
                             width=characters_per_line,
                             replace_whitespace=False,
                             drop_whitespace=False,
                             expand_tabs=False,
                             initial_indent="",
                             subsequent_indent="")
        return wrap

    # Default axes
    ax = plt.gca() if ax is None else ax
    n = len(line)

    # Compute cumulative width of line
    cumulative_width = text2cumul_width(
        text=line,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax
    )

    # Break up line
    result = []
    start_idx = 0
    c_width = 0
    while start_idx < n:

        # Determine next max break-point
        end_idx = 0
        for end_idx in range(start_idx, n + 1):
            if end_idx == n:
                break
            if cumulative_width[end_idx] - c_width > x_range:
                break
        end_idx -= 1

        # Get current line snippet and wanted width
        snippet = line[start_idx:end_idx + 1]
        wanted_width = end_idx - start_idx

        # Use textwrap to break line
        if not wanted_width or end_idx >= n - 1:
            start_idx = n
            next_line = snippet

        else:
            wrapped = textwrap.wrap(
                text=snippet,
                width=wanted_width,
                replace_whitespace=False,
                drop_whitespace=False,
                expand_tabs=False,
                initial_indent="",
                subsequent_indent="",
            )
            next_line = wrapped[0]

            # Move marker
            start_idx = start_idx + len(next_line)
            c_width = cumulative_width[start_idx]

        # Store next line
        result.append(next_line)

    return result


def flow_text_into_axes(text, x=None, y=None, fontsize=15, fontname="serif", line_spacing=1.1,
                        right_relative_margin=0.1, ax=None, fig=None, modifiers=None, verbose=False):
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
    :param fig:
    :param list[TextModifier] modifiers:  Modifiers for text.
    :param bool verbose: Want prints?
    """
    # Ensure format and copy of list
    if modifiers is not None:
        modifiers = list(sorted(deepcopy(modifiers)))
    else:
        modifiers = []

    # Default axes
    ax = plt.gca() if ax is None else ax

    # Line limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Default x and y
    if x is None:
        x = (x_max - x_min) * 0.025
    if y is None:
        y = y_max - (y_max - y_min) * 0.1

    # Determine font-size and break up the text
    fontsize, height, wrapped_text, original_newlines, additional_newlines = break_up_text_for_axes(
        text=text,
        x=x,
        y=y,
        fontsize=fontsize,
        fontname=fontname,
        line_spacing=line_spacing,
        right_relative_margin=right_relative_margin,
        ax=ax
    )
    print(fontsize)
    print(wrapped_text)
    skip_lines = list(reversed(sorted(set(additional_newlines))))
    if verbose:
        print(skip_lines)

    # Plot lines
    line_start = 0
    skips = 0
    next_newline = skip_lines.pop() if skip_lines else np.infty
    for line_nr, line in enumerate(wrapped_text):
        # Determine end of line
        line_end = line_start + len(line)

        # Add skip if an additional newline was inserted
        while next_newline < line_end - skips:
            skips += 1
            next_newline = skip_lines.pop() if skip_lines else np.infty

        # Find relevant modifiers
        relevant_modifiers = [modifier.offset(-line_start + skips) for modifier in modifiers
                              if line_start - skips <= modifier.end and modifier.start <= line_end - skips]

        # Compute y-location
        c_y = y - line_nr * height * line_spacing

        if verbose:
            print(line)
            print("\tline_start: {}".format(line_start))
            print("\tskips: {}".format(skips))
            print("\tline_end: {}".format(line_end))
            print("\trelevant_modifiers: {}".format(relevant_modifiers))

        # View line
        plot_modified_line(
            x=x,
            y=c_y,
            text=line,
            modifiers=relevant_modifiers,
            ax=ax,
            fig=fig,
            fontsize=fontsize,
        )

        # Next line
        line_start = line_end + 1


if __name__ == "__main__":
    manual_lines = False

    with Path("src", "text", "font_info", "test_text.txt").open("r") as file:
        text_lines = file.readlines()
        text_lines = text_lines[:3]

    if manual_lines:
        text_lines = [
            "",
            "\nhey there",
            "how are you?",
            "why are you asking me?!?",
            "dude I was just being polite, take a chill pill",
        ]

    the_text = "\n".join(text_lines)

    letter_modifiers = dict(
        e="blue",
        # a="red",
        # t="orange",
        # s="green",
    )

    the_modifiers = []
    for letter, color in letter_modifiers.items():
        the_modifiers.extend([
            TextModifier(val, val + 1, "color", color) for val, char in enumerate(the_text) if char == letter
        ])

    plt.close("all")
    plt.figure()
    flow_text_into_axes(text=the_text, modifiers=the_modifiers)
