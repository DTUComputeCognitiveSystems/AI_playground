import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.real_time.base_backend import BackendInterface
from src.real_time.text_input_backend import TextInputLoop
from src.text.utility.text_modifiers import TextModifier
from src.text.utility.text_plots import flow_text_into_axes


class HighlightCharacters(BackendInterface):
    def __init__(self, backend, lines_in_view=20, letter_modifiers=None):
        super().__init__()
        self.n_lines = lines_in_view
        self.backend = backend  # type: TextInputLoop
        self.letter_modifiers = dict() if letter_modifiers is None else letter_modifiers

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

    def _loop_step(self):

        # Clear axes
        self.ax.clear()

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Get current text
        text = self.backend.current_str

        # Check if there is any text
        if text:

            # Make modifiers
            modifiers = []
            for letter, color in self.letter_modifiers.items():
                modifiers.extend([
                    TextModifier(val, val + 1, "color", color) for val, char in enumerate(text) if char == letter
                ])

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
    the_backend.add_interface(HighlightCharacters(
        the_backend,
        letter_modifiers=dict(
            e=np.array((0., 0., 1.)),
            # a="red",
            # t="orange",
            # s="green",
        )
    ))
    the_backend.start()
