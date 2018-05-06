from collections import Iterable
from queue import Queue
from time import time

from ipywidgets import widgets

from src.real_time.base_backend import BackendLoop, BackendInterface


class ConsoleInputBackend(BackendLoop):
    def __init__(self, backend_interface=(), use_widget=False, n_lines=20,
                 widget_name="Input:"):
        """
        :param BackendInterface backend_interface:
        """
        self.n_lines = n_lines
        self.use_widget = use_widget

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self._current_loop_nr = self._start_time = self._str_lines = self._widget = None

        # If we are doing widgets
        if use_widget:
            self._widget = widgets.Textarea(
                value='',
                placeholder='',
                description=widget_name,
                disabled=False,
                layout=dict(width="80%")
            )
            self._widget.observe(self._widget_update, )

    def _start(self):
        # Initialize
        if self.n_lines > 0 and not self.use_widget:
            self._str_lines = Queue()
        else:
            self._str_lines = []

        # Set time of start and loop nr
        self._start_time = time()
        self._current_loop_nr = 0

        # Initialize parts through interfaces
        self.interface_loop_initialization()

        # Run console
        if self.use_widget:
            self._widget_update()
        else:
            self._console_run()

    def _widget_update(self, event=None):

        # Get text from widget
        text = self._widget.value

        # Check if text has changed (some event does not change the text)
        if text != self.current_str:

            # Break into lines and pass onto storage
            self._str_lines = []
            for line in text.split("\n"):
                self._str_lines.append(line)

            # Loop nr
            self._current_loop_nr += 1

            # Update through interface
            self.interface_loop_step()

        # TODO: Stop-checks and finalizing missing

    def _console_run(self):
        quit_flat = "$quit"

        text_indent = "\t\t"

        print("-" * 100)
        print(text_indent + "Enter/Paste your content.")
        print(text_indent + "To stop, try Ctrl-D or write: ")
        print(text_indent + "\t{}".format(quit_flat))
        print("-" * 100)

        running = True
        while running:
            try:
                lines = input()
            except (EOFError, KeyboardInterrupt):
                break

            # Check for quit
            if lines.strip() == quit_flat:
                break

            # Split into lines
            lines = lines.split("\n")

            # Store line
            for line in lines:
                if isinstance(self._str_lines, Queue):
                    self._str_lines.put(line)
                else:
                    self._str_lines.append(line)

            # If using queue remove excessive lines
            if isinstance(self._str_lines, Queue):
                while self._str_lines.qsize() > self.n_lines:
                    self._str_lines.get()

            # Loop nr
            self._current_loop_nr += 1

            # Update through interface
            self.interface_loop_step()

            # Check for end
            if self.interface_loop_stop_check():
                running = False

        # Finalize interfaces
        self.interface_finalize()

        # TODO: Also use interrupt-handler

    def start(self):
        super().start()
        return self._widget

    @property
    def widget(self):
        return self._widget

    @property
    def current_str(self) -> str:
        return "\n".join(self.string_lines)

    @property
    def string_lines(self) -> list:
        if isinstance(self._str_lines, Queue):
            return list(self._str_lines.queue)
        return self._str_lines

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time


class _TestInterface(BackendInterface):
    def __init__(self, backend, n_lines=3):
        super().__init__()
        self.n_lines = n_lines
        self.backend = backend
        self.loop_step = lambda: print("Received: {}".format(backend.current_str))

        self.loop_stop_check = lambda: backend.current_loop_nr >= self.n_lines


if __name__ == "__main__":

    the_backend = ConsoleInputBackend()
    the_backend.add_interface(_TestInterface(the_backend))
    the_backend.start()
