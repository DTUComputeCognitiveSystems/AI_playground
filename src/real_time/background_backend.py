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

            # Run animation step
            self.interface_loop_step()

    ######

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time


if __name__ == "__main__":

    ###
    # Example 1

    print("Example 1\n" + "-" * 30 + "\n")

    # Make backend
    backend = BackgroundLoop()

    # Make interface object
    stop_flags = [True] + [False for _ in range(5)]
    interface = BackendInterface(
        loop_initialization=lambda: print("before"),
        loop_step=lambda: print("iteration {}".format(backend.current_loop_nr)),
        loop_stop_check=lambda: stop_flags.pop(),
        finalize=lambda: print("after")
    )

    # Add interface adn run
    backend.add_interface(interface=interface)
    backend.start()

    ###
    # Example 2

    print("\n\nExample 2\n" + "-" * 30 + "\n")

    # This is a class that can do work at each real time iteration
    class ExampleInterface(BackendInterface):
        def __init__(self):
            self.counter = None

        # Here I initialize all variables, models, plots etc.
        def _loop_initialization(self):
            print("Initializing ExampleInterface.")
            self.counter = 0

        # Here I do all the work of one iteration
        def _loop_step(self):
            self.counter += 1
            print("Step: {}".format(self.counter))

        # Here I check whether my job is done (after which the real time loop will also stop)
        def _loop_stop_check(self):
            print("\tChecking for stop")
            if self.counter >= 10:
                print("\tStop!")
                return True
            return False

        # Here I can finalize my job, like saving results to a file, making a checkpoint etc.
        def _finalize(self):
            print("Finalizing")

        # Here I can do things if the loop is interrupted (for example by the user). An idea could be to write a log.
        def _interrupt_handler(self):
            print("Interrupted!")

        # Here I say how often I would like to run. The loop will run according to the "slowest" interface (bottleneck).
        @property
        def loop_time_milliseconds(self):
            return 400

    # Making a backend, making an interface, adding the interface and running the loop
    backend = BackgroundLoop()
    interface = ExampleInterface()
    backend.add_interface(interface=interface)
    backend.start()
