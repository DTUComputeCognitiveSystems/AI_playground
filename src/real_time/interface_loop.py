# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

# pylint: disable=E0202
class LoopInterface:
    """
    Interface for a backend. 5 Functions can be set for performing tasks at various relevant times.
    The wanted loop-time can also be set.
    """
    def __init__(self, loop_initialize=noop, loop_step=noop, loop_stop_check=noop, loop_finalize=noop,
                 loop_interrupt_handler=noop, loop_time_milliseconds=200):
        """
        Dynamic version of the backend interface.
        :param Callable loop_initialization:
        :param Callable loop_step:
        :param Callable loop_stop_check:
        :param Callable finalize:
        :param Callable loop_interrupt_handler:
        :param int loop_time_milliseconds:
        """
        self._loop_initialize = loop_initialize
        self._loop_step = loop_step
        self._loop_stop_check = loop_stop_check
        self._loop_finalize = loop_finalize
        self._loop_interrupt_handler = loop_interrupt_handler
        self._loop_time_milliseconds = loop_time_milliseconds

    def _loop_initialize(self):
        pass

    @property
    def loop_initialize(self):
        return self._loop_initialize

    def _loop_step(self):
        pass

    @property
    def loop_step(self):
        return self._loop_step

    def _loop_stop_check(self):
        pass

    @property
    def loop_stop_check(self):
        return self._loop_stop_check

    def _loop_finalize(self):
        pass

    @property
    def loop_finalize(self):
        return self._loop_finalize

    def _loop_interrupt_handler(self):
        pass

    @property
    def loop_interrupt_handler(self):
        return self._loop_interrupt_handler

    def _loop_time_milliseconds(self):
        pass

    @property
    def loop_time_milliseconds(self):
        return self._loop_time_milliseconds