
# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

#class BackendInterfaceObject(BackendInterface):

class BackendInterface:
    """
    Interface for a backend. 5 Functions can be set for performing tasks at various relevant times.
    The wanted loop-time can also be set.
    """
    def __init__(self, loop_initialization=noop, loop_step=noop, loop_stop_check=noop, finalize=noop,
                 interrupt_handler=noop, loop_time_milliseconds=200):
        """
        Dynamic version of the backend interface.
        :param Callable loop_initialization:
        :param Callable loop_step:
        :param Callable loop_stop_check:
        :param Callable finalize:
        :param Callable interrupt_handler:
        :param int loop_time_milliseconds:
        """
        self._loop_initialization = loop_initialization
        self._loop_step = loop_step
        self._loop_stop_check = loop_stop_check
        self._finalize = finalize
        self._interrupt_handler = interrupt_handler
        self._loop_time_milliseconds = loop_time_milliseconds
    def _loop_initialization(self):
        pass

    @property
    def loop_initialization(self):
        return self._loop_initialization

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

    def _finalize(self):
        pass

    @property
    def finalize(self):
        return self._finalize

    def _interrupt_handler(self):
        pass

    @property
    def interrupt_handler(self):
        return self._interrupt_handler

    @property
    def loop_time_milliseconds(self):
        pass


class BackendMultiInterface:
    def __init__(self, *args):
        """
        Interface for a backend. Multiple interfaces can be combined in this class and will all be run in the
        real-time loop.
        :param tuple args: Interfaces.
        """
        self.interfaces = list(args)  # type: list[BackendInterface]
        self._loop_time_milliseconds = None

    def add_interface(self, interface: BackendInterface):
        self.interfaces.append(interface)
        self._loop_time_milliseconds = max([interface.loop_time_milliseconds for interface in self.interfaces])

    def __len__(self):
        return len(self.interfaces)

    def __bool__(self):
        return bool(self.interfaces)

    def _loop_initialization(self):
        for interface in self.interfaces:
            interface.loop_initialization()

    @property
    def loop_initialization(self):
        return self._loop_initialization

    def _loop_step(self):
        for interface in self.interfaces:
            interface.loop_step()

    @property
    def loop_step(self):
        return self._loop_step

    def _loop_stop_check(self):
        return any([interface.loop_stop_check() for interface in self.interfaces])

    @property
    def loop_stop_check(self):
        return self._loop_stop_check

    def _finalize(self):
        for interface in self.interfaces:
            interface.finalize()

    @property
    def finalize(self):
        return self._finalize

    def _interrupt_handler(self):
        for interface in self.interfaces:
            interface.interrupt_handler()

    @property
    def interrupt_handler(self):
        return self._interrupt_handler

    @property
    def loop_time_milliseconds(self):
        if self._loop_time_milliseconds is None:
            self._loop_time_milliseconds = max([interface.loop_time_milliseconds for interface in self.interfaces])
        return self._loop_time_milliseconds


class BackendLoop:
    def __init__(self, *args):
        self.interface = BackendMultiInterface(*args)

        # Defaults
        self.stop_now = False

    def add_interface(self, interface):
        self.interface.add_interface(interface=interface)

    def start(self):
        if not self.interface:
            raise ValueError("No interfaces given to backend. We have nothing to run :/")
        self._start()

    def _start(self):
        raise NotImplementedError

    @property
    def current_loop_nr(self) -> int:
        raise NotImplementedError

    @property
    def start_time(self) -> float:
        raise NotImplementedError

    # Interface redirect

    @property
    def interface_loop_initialization(self):
        return self.interface.loop_initialization

    @property
    def interface_loop_step(self):
        return self.interface.loop_step

    @property
    def interface_loop_stop_check(self):
        return self.interface.loop_stop_check

    @property
    def interface_finalize(self):
        return self.interface.finalize

    @property
    def interface_interrupt_handler(self):
        return self.interface.interrupt_handler

    @property
    def loop_time_milliseconds(self):
        return self.interface.loop_time_milliseconds
