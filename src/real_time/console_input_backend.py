from _tkinter import TclError
from collections import Iterable
from time import time

from src.real_time.base_backend import BackendLoop, BackendInterface


class ConsoleInputBackend(BackendLoop):
    def __init__(self, backend_interface=()):
        """
        :param BackendInterface backend_interface:
        """

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self._current_loop_nr = self._start_time = self._input_lines = self._current_str = None

    def _start(self):
        # Initialize
        self._input_lines = []

        # Set time of start and loop nr
        self._start_time = time()
        self._current_loop_nr = 0

        # Initialize parts through interfaces
        self.interface_loop_initialization()

        quit_flat = "$quit"

        text_indent = "\t\t"

        print("-" * 100)
        print(text_indent + "Enter/Paste your content.")
        print(text_indent + "To stop, try Ctrl-D or write: ")
        print(text_indent + "\t{}".format(quit_flat))
        print("-" * 100)

        while True:
            line = ""
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break

            # Check for quit
            if line.strip() == quit_flat:
                break

            # Store line
            self._input_lines.append(line)
            self._current_str = line

            # Update through interface
            self.interface_loop_step()

            # TODO: Consider using interface_loop_stop_check()

        # Finalize interfaces
        self.interface_finalize()

        # TODO: Also use interrupt-handler

    @property
    def current_str(self) -> str:
        return self._current_str

    @property
    def string_lines(self) -> list:
        return self._input_lines

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
