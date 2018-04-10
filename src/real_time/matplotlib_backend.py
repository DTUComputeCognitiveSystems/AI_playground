from _tkinter import TclError
from time import time

from matplotlib import pyplot as plt, animation

from src.real_time.base_backend import ProgramLoop, noop


class MatplotlibProgramLoop(ProgramLoop):
    def __init__(self, loop_time_milliseconds=200,
                 title="Real time animation",
                 fig=None, block=True, blit=False):
        # Settings
        self._blit = blit
        self._title = title
        self._block = block
        self.loop_time_milliseconds = loop_time_milliseconds
        self._fig = fig
        self.stop_now = False

        # Set real-time loop functions
        super().__init__(noop, noop, self.__stop_animation, noop, noop)

        # For holding artists for Matplotlib
        self.artists = []

        # Fields
        self._canvas = self._ax = self._animation = self.frame_nr = None

    def start(self):
        # Default figure
        if self._fig is None:
            self._fig = plt.figure()

        # Set canvas
        self._canvas = self._fig.canvas
        plt.title(self._title)
        self._canvas.set_window_title(self._title)

        # Get axes
        self._ax = plt.gca()

        # Do animation
        self._animation = animation.FuncAnimation(
            fig=self._fig,
            func=self.__animate_step,
            init_func=self.__initialize_animation,
            interval=self.loop_time_milliseconds,
            repeat=False,
            frames=None,
            blit=self._blit
        )

        # Block main thread
        if self._block:
            self.__wait_for_end()
            plt.close(self._fig)

    # noinspection PyUnusedLocal
    def __stop_animation(self, *args, **kwargs):
        if time() > self.start_time + 10:
            return True
        return False

    def __initialize_animation(self):

        # Allow additional artists from child classes
        self.loop_initialization(self)

        # Make sure figure is drawn
        plt.draw()
        plt.show()

        # Set time of start
        self.start_time = time()

        return self.artists

    def __animate_step(self, i):
        self.frame_nr = i

        # Run animation step
        self.loop_step(self)

        # Check for end
        if self.loop_stop_check(self):
            self._fig.canvas.stop_event_loop()
            self.__finalize()
            self.stop_now = True

        return self.artists

    def __finalize(self):
        self.finalize()
        plt.close("all")

    def __wait_for_end(self):
        try:
            while not self.stop_now:
                plt.pause(0.2)
        except (TclError, KeyboardInterrupt):
            plt.close("all")
            self.interrupt_handler()