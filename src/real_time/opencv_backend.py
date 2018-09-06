from _tkinter import TclError
from collections import Iterable
from time import time

import cv2
from src.real_time.base_backend import BackendLoop, BackendInterfaceObject


class OpenCVLoop(BackendLoop):
    def __init__(self, backend_interface=(),
                 title="Real time animation", 
                 block = False):
        """
        :param BackendInterfaceObject backend_interface:
        :param str title:
        """
        # Settings
        self.title = title
        self.block = block

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self._current_loop_nr = 0
        self._start_time = None

    def _start(self):

        # Do loop
        # Initialize parts through interfaces
        self.interface_loop_initialization()
        # Set time of start
        self._start_time = time()

        while self.stop_now == False:
            self._current_loop_nr += 1

            # Run loop step
            self.interface_loop_step()

            # Check for end
            if self.interface_loop_stop_check() or self.stop_now or cv2.getWindowProperty(self.title, 0) < 0 or cv2.waitKey(self.loop_time_milliseconds) == 27:
                self.__finalize()
                self.stop_now = True

        # Block main thread
        if self.block:
            self.__wait_for_end()
            # Finalize loop
            self.interface_finalize()
            cv2.destroyAllWindows()

    def __finalize(self):
        self.interface_finalize()
        cv2.destroyAllWindows()

    def __wait_for_end(self):
        try:
            while not self.stop_now:
                cv2.waitKey(self.loop_time_milliseconds)
        except (TclError, KeyboardInterrupt):
            cv2.destroyAllWindows()
            self.interface_interrupt_handler()

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
