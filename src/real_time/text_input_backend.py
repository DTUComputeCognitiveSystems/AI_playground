from collections import Iterable
from queue import Queue, Empty
from threading import Thread, Event, Timer
from time import time

from ipywidgets import widgets
from IPython.display import display

from src.real_time.base_backend import BackendLoop, BackendInterface


class UpdateChecker(Thread):
    def __init__(self, in_queue, out_queue, main_thread_ready, queue_timout):
        super().__init__()
        self.queue_timout = queue_timout

        # Events and queues
        self.main_thread_ready = main_thread_ready  # type: Event
        self.in_queue = in_queue  # type: Queue
        self.out_queue = out_queue  # type: Queue

        # For holding the current value
        self.value = None

        # Timer and and flag for sending data
        self.timer = None
        self.needs_updating = False

    def run(self):
        while True:
            try:
                # Get data and set flag for updating if data is accessible
                self.value = self.in_queue.get(timeout=self.queue_timout)
                self.needs_updating = True
            except Empty:
                pass

            # Check if there is data to process and whether the main-process is ready
            if self.needs_updating and self.main_thread_ready.is_set():
                # Send data
                self.out_queue.put(self.value)

                # Notify main thread
                self.main_thread_ready.clear()

                # No longer needs updating
                self.needs_updating = False


class TextInputLoop(BackendLoop):
    def __init__(self, backend_interface=(), use_widget=False, n_lines=20,
                 widget_name="Input:", check_delay=.15, text_height="450px"):
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
            self.check_delay = check_delay

            # Event for whether the main thread is ready to process data
            self.main_thread_ready = Event()
            self.main_thread_ready.set()

            # Make queues for sending and receiving data
            self.ch_in_queue = Queue()
            self.ch_out_queue = Queue()

            # Make checker for ensuring what data is processed
            self.checker = UpdateChecker(in_queue=self.ch_in_queue, out_queue=self.ch_out_queue,
                                         main_thread_ready=self.main_thread_ready,
                                         queue_timout=self.check_delay)
            self.checker.daemon = True
            self.checker.start()

            # Set a timer for checking output
            self.timer = Timer(check_delay, self.check_output)
            self.timer.start()

            self._widget_label = widgets.Label(
                value=widget_name,
                margin="0px 0px 0px 0px",
            )
            self._widget = widgets.Textarea(
                value='',
                placeholder='',
                description="",
                disabled=False,
                layout=dict(width="90%", height=text_height, margin="-3px 0px 0px 0px"),
            )
            self._widget.observe(self._widget_update, )

    def check_output(self):
        # If main-thread has been asked to process data
        if not self.main_thread_ready.is_set():
            # Get data
            text = self.ch_out_queue.get()

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

            # Main thread is once again ready for data
            self.main_thread_ready.set()

        # Set a time for checking for data
        self.timer = Timer(self.check_delay, self.check_output)
        self.timer.start()

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

    def _widget_update(self, _=None):
        # Get text from widget
        text = self._widget.value

        # Send to thread
        self.ch_in_queue.put(text)

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
        self.display()

    # noinspection PyTypeChecker
    def display(self):
        display(self._widget_label)
        display(self._widget)

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


class _TestInterfaceOLD(BackendInterface):
    def __init__(self, backend, n_lines=3):
        super().__init__()
        self.n_lines = n_lines
        self.backend = backend
        self._loop_step = lambda: print("Received: {}".format(backend.current_str))

        self._loop_stop_check = lambda: backend.current_loop_nr >= self.n_lines


if __name__ == "__main__":

    the_backend = TextInputLoop()
    the_backend.add_interface(_TestInterfaceOLD(the_backend))
    the_backend.start()
