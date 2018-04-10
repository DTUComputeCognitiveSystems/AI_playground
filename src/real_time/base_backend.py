# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass


class ProgramLoop:
    def __init__(self, loop_initialization, loop_step, loop_stop_check, finalize,
                 interrupt_handler):
        self.loop_initialization = loop_initialization
        self.loop_step = loop_step
        self.loop_stop_check = loop_stop_check
        self.finalize = finalize
        self.interrupt_handler = interrupt_handler

    def start(self):
        raise NotImplementedError
