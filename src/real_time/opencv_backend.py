from _tkinter import TclError
from collections import Iterable
from time import time

import cv2
from src.real_time.base_backend import BackendLoop, BackendInterface


class OpenCVLoop(BackendLoop):
    def __init__(self, backend_interface=(),
                 title = "Real time webcam stream", opencv_frame = None):
        """
        :param BackendInterface backend_interface:
        :param str title:
        """
        # Settings
        self.title = title

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self._current_loop_nr = None
        self._start_time = None

    def _start(self):

        # Do loop
        self._current_loop_nr = 0
        # Initialize parts through interfaces
        self.interface_loop_initialization()
        # Set time of start
        self._start_time = time()

        while self.stop_now == False:
            self._current_loop_nr += 1
 
            # Run loop step
            self.interface_loop_step()
            key = cv2.waitKey(self.loop_time_milliseconds)
            
            # Check for end
            if self.interface_loop_stop_check() or self.stop_now or key == 27 or cv2.getWindowProperty(self.title, 0) == -1:
                #print("stopcondition {}, {}, {}".format(self.interface_loop_stop_check(), self.stop_now, cv2.getWindowProperty(self.title, 0)))
                self.interface_finalize()
                cv2.destroyAllWindows()
                self.stop_now = True 

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
