from _tkinter import TclError
from collections import Iterable
from time import time

from matplotlib import pyplot as plt, animation

from src.real_time.base_backend import BackendLoop, BackendInterfaceObject


class MatplotlibLoop(BackendLoop):
    def __init__(self, backend_interface=(),
                 title="Real time animation",
                 fig=None, block=False, blit=False):
        """
        :param BackendInterfaceObject backend_interface:
        :param str title:
        :param fig:
        :param bool block:
        :param bool blit:
        """
        # Settings
        self.blit = blit
        self.title = title
        self.block = block
        self.fig = fig

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self.artists = []
        self.canvas = self.ax = self._animation = self._current_loop_nr = self._start_time = None

    def _start(self):
        # Default figure
        if self.fig is None:
            self.fig = plt.figure()

        # Set canvas
        self.canvas = self.fig.canvas
        plt.title(self.title)
        self.canvas.set_window_title(self.title)

        # Set close-event
        def closer(_):
            self.stop_now = True
        self.canvas.mpl_connect('close_event', closer)

        # Do animation
        self._animation = animation.FuncAnimation(
            fig=self.fig,
            func=self.__animate_step,
            init_func=self.__initialize_animation,
            interval=self.loop_time_milliseconds,
            repeat=False,
            frames=None,
            blit=self.blit
        )

        # Block main thread
        if self.block:
            self.__wait_for_end()
            plt.close(self.fig)

    def __initialize_animation(self):

        # Initialize parts through interfaces
        self.interface_loop_initialization()

        # Make sure figure is drawn
        plt.draw()
        plt.show()

        # Set time of start
        self._start_time = time()

        return self.artists

    def __animate_step(self, i):
        self._current_loop_nr = i

        # Run animation step
        self.interface_loop_step()

        # Check for end
        if self.interface_loop_stop_check() or self.stop_now:
            self.fig.canvas.stop_event_loop()
            self.__finalize()
            self.stop_now = True

        return self.artists

    def __finalize(self):
        self.interface_finalize()
        plt.close("all")

    def __wait_for_end(self):
        try:
            while not self.stop_now:
                plt.pause(0.2)
        except (TclError, KeyboardInterrupt):
            plt.close("all")
            self.interface_interrupt_handler()

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
