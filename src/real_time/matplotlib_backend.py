from _tkinter import TclError
from time import time

from matplotlib import pyplot as plt, animation

from src.real_time.base_backend import BackendLoop, BackendInterface


class MatplotlibLoop(BackendLoop):
    def __init__(self, backend_interface,
                 title="Real time animation",
                 fig=None, block=True, blit=False):
        """
        :param BackendInterface backend_interface:
        :param str title:
        :param fig:
        :param bool block:
        :param bool blit:
        """
        # Settings
        self._blit = blit
        self._title = title
        self._block = block
        self._fig = fig

        # Connect interface
        super().__init__(backend_interface)

        # Fields
        self.canvas = self.ax = self._animation = self._current_loop_nr = self._start_time = None

    def start(self):
        # Default figure
        if self._fig is None:
            self._fig = plt.figure()

        # Set canvas
        self.canvas = self._fig.canvas
        plt.title(self._title)
        self.canvas.set_window_title(self._title)

        # Set close-event
        def closer(_):
            self.stop_now = True
        self.canvas.mpl_connect('close_event', closer)

        # Get axes
        self.ax = plt.gca()

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

    def __initialize_animation(self):

        # Allow additional artists from child classes
        self.loop_initialization()

        # Make sure figure is drawn
        plt.draw()
        plt.show()

        # Set time of start
        self._start_time = time()

        return self.artists

    def __animate_step(self, i):
        self._current_loop_nr = i

        # Run animation step
        self.loop_step()

        # Check for end
        if self.loop_stop_check():
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

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
