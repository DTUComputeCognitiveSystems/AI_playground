from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.real_time.base_backend import BackendInterface, BackendLoop
from src.real_time.text_input_backend import TextInputLoop
from src.text.utility.text_modifiers import TextModifier
from src.text.utility.text_plots import flow_text_into_axes

# pylint: disable=E0202
class HighlightCharacters(BackendInterface):
    def __init__(self, backend, lines_in_view=20, letter_modifiers=None):
        super().__init__(
            loop_initialization = self._loop_initialization,
            loop_step = self._loop_step,
            loop_stop_check = self._loop_stop_check,
            finalize = self._finalize,
            interrupt_handler = self._interrupt_handler,
            loop_time_milliseconds = self.loop_time_milliseconds
        )
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


class _TextExampleBackend(BackendLoop):
    def __init__(self, lines, delay=1.0, do_finalize=False, *args):
        super().__init__(*args)
        self.do_finalize = do_finalize
        self.delay = delay
        self.lines = lines
        self.line_nr = None
        self.line = None
        self.current_str = None

    def _start(self):
        self.interface_loop_initialization()
        self._start_time = time()
        self.current_str = ""

        for self.line_nr, self.line in enumerate(self.lines):
            self.current_str += "\n" + self.line
            self.interface_loop_step()

            sleep(self.delay)

        if self.do_finalize:
            self.interface_finalize()

    @property
    def current_loop_nr(self) -> int:
        return self.line_nr

    @property
    def start_time(self) -> float:
        return self._start_time


if __name__ == "__main__":

    text_lines = [
        "hey there",
        "how are you doing?",
        "why are you asking me?!?",
        "dude I was just being polite, take a chill pill",
    ]

    the_backend = _TextExampleBackend(lines=text_lines)

    plt.close("all")
    the_backend.add_interface(HighlightCharacters(
        the_backend,
        letter_modifiers=dict(
            e="blue",
            a="red",
            t="orange",
            s="green",
        )
    ))
    the_backend.start()
