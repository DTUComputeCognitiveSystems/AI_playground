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


def _sentiment_format(sentiment):
    # Get sign
    sign = np.sign(sentiment)

    # Determine formatting
    weight, style, color_val = _sentiment_styles[int(abs(sentiment))]
    color_val = sign * color_val

    # Compute color
    color = np.array([0., 0., 0.])
    color[0] -= min(color_val, 0)
    color[2] += max(color_val, 0)

    return color, weight, style


class SentimentHighlighter(BackendInterface):
    def __init__(self, backend, lines_in_view=20):
        super().__init__()
        self.n_lines = lines_in_view
        self.backend = backend  # type: TextInputLoop
        self.afinn = Afinn()
        self.resized_timer = None

    def _note_resize(self, _=None):
        if self.resized_timer is not None:
            self.resized_timer.cancel()
        self.resized_timer = Timer(1.0, self._loop_step)
        self.resized_timer.start()

    def _loop_initialization(self):
        self.fig = plt.figure()  # type: Figure
        self.canvas = self.fig.canvas
        self.ax = plt.gca()

        # Remove ticks and coordinates
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.format_coord = lambda x, y: ''

        # Draw canvas
        self.canvas.draw()
        plt.pause(0.05)

        # If canvas is resised - then redraw
        self.canvas.mpl_connect("resize_event", self._note_resize)

    def _loop_step(self, _=None):
        self.resized_timer = None

        # Clear axes
        self.ax.clear()

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Get current text
        text = self.backend.current_str

        # Check if there is any text
        if text:

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
                color, weight, style = _sentiment_format(sentiment=sentiment)

                # Add modifier
                modifiers.append(TextModifier(idx, end, "color", color))
                if weight is not None:
                    modifiers.append(TextModifier(idx, end, "weight", weight))
                if style is not None:
                    modifiers.append(TextModifier(idx, end, "style", style))

                # Next
                idx = end

            # Plots text
            flow_text_into_axes(
                fig=self.fig,
                ax=self.ax,
                text=text,
                modifiers=modifiers
            )

        # Draw canvas
        self.canvas.draw()
        plt.pause(0.05)

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
    ))
    the_backend.start()
