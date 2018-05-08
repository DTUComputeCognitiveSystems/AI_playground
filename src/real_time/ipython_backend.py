import sched
from collections import Iterable
from time import time

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from src.real_time.base_backend import BackendLoop, BackendInterfaceObject


class IPythonLoop(BackendLoop):
    def __init__(self, backend_interface=(), fig=None, title="Real time animation",
                 block=True):
        """
        :param BackendInterfaceObject backend_interface:
        :param bool block:
        """
        # Settings
        self.title = title
        self.fig = fig
        self.block = block

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self.artists = []
        self.widgets = []
        self.canvas = self.scheduler = self._start_time = self._current_loop_nr = self.ax = None

    def _start(self):
        # Default figure
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        elif isinstance(self.fig, tuple):
            self.fig = plt.figure(figsize=self.fig)
            self.ax = plt.gca()

        # Set canvas
        self.canvas = self.fig.canvas
        plt.title(self.title)
        self.canvas.set_window_title(self.title)

        # Set close-event
        def closer(_):
            self.stop_now = True
        self.canvas.mpl_connect('close_event', closer)

        # Set scheduler
        self.scheduler = sched.scheduler()

        # Initialize parts through interfaces
        self.interface_loop_initialization()
        clear_output(wait=True)

        # Set time of start
        self._start_time = time()

        # Loop number
        self._current_loop_nr = 0

        # Display now
        display(self.fig)

        # Set starting event
        self.scheduler.enter(
            delay=self.loop_time_milliseconds / 1000.0,
            priority=1,
            action=self._loop_step,
        )

        # Start loop
        try:
            self.scheduler.run(blocking=self.block)
        except KeyboardInterrupt:
            self.interface_interrupt_handler()

        # Wait
        plt.show(block=True)

    def _loop_step(self):

        self._current_loop_nr += 1

        # Run animation step
        self.interface_loop_step()

        # Check for end
        if self.interface_loop_stop_check() or self.stop_now:
            self.interface_finalize()
            self.stop_now = True

        # Otherwise set next event
        else:
            self.scheduler.enter(
                delay=self.loop_time_milliseconds / 1000.0,
                priority=1,
                action=self._loop_step,
            )
        
        clear_output(wait=True)
        display(self.fig)

    ######

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
