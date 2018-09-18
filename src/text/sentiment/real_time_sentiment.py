from copy import deepcopy
from threading import Timer

import matplotlib.pyplot as plt
from afinn import Afinn
from matplotlib.figure import Figure

from src.real_time.base_backend import BackendInterface
from src.real_time.text_input_backend import TextInputLoop
from src.text.sentiment.sentiment_highlighting import sentiment_text_modifiers
from src.text.utility.text_plots import flow_text_into_axes


class SentimentHighlighter(BackendInterface):
    def __init__(self, backend, lines_in_view=20, remove_axes=True, facecolor='white'):
        super().__init__(
            loop_initialization = self._loop_initialization,
            loop_step = self._loop_step,
            loop_stop_check = self._loop_stop_check,
            finalize = self._finalize,
            interrupt_handler = self._interrupt_handler,
            loop_time_milliseconds = self.loop_time_milliseconds
        )
        self.facecolor = facecolor
        self.remove_axes = remove_axes
        self.n_lines = lines_in_view
        self.backend = backend  # type: TextInputLoop
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
                self.c_modifiers = sentiment_text_modifiers(text=self.c_text, full_contrast=self.full_contrast)
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
