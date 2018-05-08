import matplotlib.pyplot as plt
from math import sin

from src.image.video.snapshot import VideoCamera
from src.real_time.base_backend import BackendInterfaceObject
from src.real_time.matplotlib_backend import MatplotlibLoop
from src.real_time.base_backend import BackendLoop


class SineCurve:
    def __init__(self, backend: BackendLoop, title="Sine curve", ax=None, time_per_frame=0.5):
        self._title = title
        self.ax = ax if ax is not None else plt.gca()
        self._c_time = self.curve = None
        self.xs = None  # type: list
        self.ys = None  # type: list
        self.time_delta = time_per_frame

        # Make real-time interface
        interface = BackendInterfaceObject(
            loop_initialization=self._loop_initialization,
            loop_step=self._loop_step,
            loop_stop_check=self._loop_stop_check,
            finalize=self._finalize,
            interrupt_handler=self._interrupt_handler,
        )
        backend.add_interface(interface=interface)

    def _loop_initialization(self):
        plt.sca(self.ax)
        self.ax.set_title(self._title)
        self.curve, = plt.plot([1], [1])
        self._c_time = 0
        self.xs = []
        self.ys = []

    def _loop_step(self):
        self.xs.append(self._c_time)
        self.ys.append(sin(self._c_time))
        self._c_time += self.time_delta
        self.curve.set_data([self.xs, self.ys])
        self.ax.set_xlim([min(self.xs), max(self.xs) + self.time_delta])
        self.ax.set_ylim([-1, 1])

    def _loop_stop_check(self):
        return False

    def _finalize(self):
        pass

    def _interrupt_handler(self):
        pass

    def _frame_time(self):
        pass


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    figure = plt.figure()
    back_end = MatplotlibLoop(fig=figure)
    back_end.block = True

    ax1 = plt.subplot(2, 1, 1)
    the_video = VideoCamera(
        backend=back_end,
        ax=ax1
    )

    ax2 = plt.subplot(2, 1, 2)
    the_sine = SineCurve(
        backend=back_end,
        ax=ax2
    )

    back_end.start()
