from collections import Callable


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass


class BackendInterface:
    def __init__(self, loop_initialization=noop, loop_step=noop, loop_stop_check=noop, finalize=noop,
                 interrupt_handler=noop, loop_time_milliseconds=200):
        """
        Interface for a backend. 5 Functions can be set for performing tasks at various relevant times.
        Also the loop-time can be set.
        :param Callable loop_initialization:
        :param Callable loop_step:
        :param Callable loop_stop_check:
        :param Callable finalize:
        :param Callable interrupt_handler:
        :param int loop_time_milliseconds:
        """
        self.loop_initialization = loop_initialization
        self.loop_step = loop_step
        self.loop_stop_check = loop_stop_check
        self.finalize = finalize
        self.interrupt_handler = interrupt_handler
        self.loop_time_milliseconds = loop_time_milliseconds


class BackendLoop:
    def __init__(self, interface: BackendInterface):
        self.interface = interface

        # Defaults
        self.stop_now = False
        self.artists = []

    def start(self):
        raise NotImplementedError

    @property
    def current_loop_nr(self) -> int:
        raise NotImplementedError

    @property
    def start_time(self) -> float:
        raise NotImplementedError

    # Interface redirect

    @property
    def loop_initialization(self):
        return self.interface.loop_initialization

    @property
    def loop_step(self):
        return self.interface.loop_step

    @property
    def loop_stop_check(self):
        return self.interface.loop_stop_check

    @property
    def finalize(self):
        return self.interface.finalize

    @property
    def interrupt_handler(self):
        return self.interface.interrupt_handler

    @property
    def loop_time_milliseconds(self):
        return self.interface.loop_time_milliseconds


