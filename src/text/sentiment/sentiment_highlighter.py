from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from afinn import Afinn
from matplotlib.figure import Figure

from threading import Timer

from src.real_time.base_backend import BackendInterface
from src.real_time.text_input_backend import TextInputLoop
from src.text.utility.text_modifiers import TextModifier
from src.text.utility.text_plots import flow_text_into_axes

_sentiment_max = 5


_sentiment_styles = {
    1: (None, None, 0.5),
    2: (None, None, 0.75),
    3: (None, None, 1.0),
    4: (None, "italic", 1.0),
    5: ("bold", "italic", 1.0),
}


__positive_test_words = "yes sweet great fantastic superb"
__negative_test_words = "no damn bad fraud prick"


def _sentiment_format(sentiment, full_contrast):
    # Get sign
    sign = np.sign(sentiment)

    # Determine formatting
    if full_contrast:
        weight = "bold"
        style = None
        color_val = 1.0
    else:
        weight, style, color_val = _sentiment_styles[int(abs(sentiment))]
    color_val = sign * color_val

    # Compute color
    color = np.array([0., 0., 0.])
    color[0] -= min(color_val, 0)
    color[2] += max(color_val, 0)

    return color, weight, style


class SentimentHighlighter(BackendInterface):
    def __init__(self, backend, lines_in_view=20, remove_axes=True, facecolor='white'):
        super().__init__()
        self.facecolor = facecolor
        self.remove_axes = remove_axes
        self.n_lines = lines_in_view
        self.backend = backend  # type: TextInputLoop
        self.afinn = Afinn()
        self.resized_timer = None

        # For options in widgets
        self.full_contrast = False
        self.do_highlighting = True

        self.c_text = None
        self.c_modifiers = None

    def _note_resize(self, _=None):
        if self.resized_timer is not None:
            self.resized_timer.cancel()
        self.resized_timer = Timer(1.0, self._loop_step)
        self.resized_timer.start()

    def _loop_initialization(self):
        self.fig = plt.figure(facecolor=self.facecolor)  # type: Figure
        self.canvas = self.fig.canvas
        self.ax = plt.gca()

        # Remove ticks and coordinates
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.format_coord = lambda x, y: ''
        plt.tight_layout()
        if self.remove_axes:
            self.ax.set_axis_off()

        # Draw canvas
        self.canvas.draw()
        plt.pause(0.05)

        # If canvas is resised - then redraw
        self.canvas.mpl_connect("resize_event", self._note_resize)

    def make_modifiers(self, text):
        # Get sentiment-words and their sentiments
        sentiment_words = self.afinn.find_all(text)
        sentiments = [self.afinn.score(word) for word in sentiment_words]

        # Make modifiers
        modifiers = []
        idx = 0
        for word, sentiment in zip(sentiment_words, sentiments):

            # Next index
            idx = text[idx:].find(word) + idx

            # End position of word
            end = idx + len(word)

            # Determine format
            color, weight, style = _sentiment_format(sentiment=sentiment, full_contrast=self.full_contrast)

            # Add modifier
            modifiers.append(TextModifier(idx, end, "color", color))
            if weight is not None:
                modifiers.append(TextModifier(idx, end, "weight", weight))
            if style is not None:
                modifiers.append(TextModifier(idx, end, "style", style))

            # Next
            idx = end

        return modifiers

    def _loop_step(self, _=None):
        self.resized_timer = None

        # Clear axes
        self.ax.clear()

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()
        if self.remove_axes:
            self.ax.set_axis_off()

        # Get current text
        self.c_text = self.backend.current_str

        # Check if there is any text
        if self.c_text:

            # Make modifiers
            if self.do_highlighting:
                self.c_modifiers = self.make_modifiers(text=self.c_text)
            else:
                self.c_modifiers = []

            # Plots text
            flow_text_into_axes(
                fig=self.fig,
                ax=self.ax,
                text=self.c_text,
                modifiers=deepcopy(self.c_modifiers),
            )

        # Draw canvas
        self.canvas.draw()
        plt.pause(0.05)

    def refresh(self):
        self._loop_step()

    def _loop_stop_check(self):
        pass

    def _finalize(self):
        plt.close(self.fig.number)

    def _interrupt_handler(self):
        pass

    @property
    def loop_time_milliseconds(self):
        return None


if __name__ == "__main__":

    plt.close("all")
    the_backend = TextInputLoop()
    the_backend.add_interface(SentimentHighlighter(
        the_backend,
        facecolor="blue",
    ))
    the_backend.start()
