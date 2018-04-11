import sched
from collections import Iterable
from time import time

from src.real_time.base_backend import BackendLoop, BackendInterface


class BackgroundLoop(BackendLoop):
    def __init__(self, backend_interface=(),
                 block=True):
        """
        :param BackendInterface backend_interface:
        :param bool block:
        """
        # Settings
        self.block = block

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self.scheduler = self._start_time = self._current_loop_nr = None

    def _start(self):
        self.scheduler = sched.scheduler()

        # Initialize parts through interfaces
        self.interface_loop_initialization()

        # Set time of start
        self._start_time = time()

        # Loop number
        self._current_loop_nr = 0

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

    ######

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time


if __name__ == "__main__":
    counter = 0

    def print_step():
        global counter
        counter += 1
        print("Step: {}".format(counter))

    def stop_checker():
        print("\tChecking for stop")
        global counter
        if counter > 10:
            print("\tStop!")
            return True
        return False

    interface = BackendInterface(
        loop_initialization=lambda: print("Initializing interfaces."),
        loop_step=print_step,
        loop_stop_check=stop_checker,
        finalize=lambda: print("Finalizing"),
        interrupt_handler=lambda: print("Interrupted!"),
        loop_time_milliseconds=400
    )

    backend = BackgroundLoop()
    backend.add_interface(interface=interface)
    backend.start()
