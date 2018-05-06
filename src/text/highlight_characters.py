from matplotlib.figure import Figure

from src.real_time.base_backend import BackendInterfaceClass
from src.real_time.text_input_backend import ConsoleInputBackend
import matplotlib.pyplot as plt

from src.text.utility.text_modifiers import TextModifier
from src.text.utility.text_plots import flow_text_into_axes


# TODO: Make the system use a thread for maintaining the figure and ignore consequtive events (just grab the slast one)
# TODO:     I'm thinking some kind of event-queue.


class HighlightCharacters(BackendInterfaceClass):
    def __init__(self, backend, lines_in_view=20, letter_modifiers=None):
        super().__init__()
        self.n_lines = lines_in_view
        self.backend = backend  # type: ConsoleInputBackend
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
        pass

    def _interrupt_handler(self):
        pass

    @property
    def loop_time_milliseconds(self):
        return None


if __name__ == "__main__":

    plt.close("all")
    the_backend = ConsoleInputBackend()
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
