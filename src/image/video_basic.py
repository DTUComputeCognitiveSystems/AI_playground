import matplotlib

matplotlib.use('TkAgg')

from time import time

import numpy as np
from matplotlib import pyplot as plt, animation

from src.image.capture_webcam import get_photo


class VideoTexter:
    def __init__(self, backgroundcolor="darkblue", color="white", position="sw"):
        position = position.lower().strip()

        # Vertical
        if position in ("sw", "s", "se"):
            self._verticalalignment = 'bottom'
            self._y = 0.0
        elif position in ("w", "e"):
            self._verticalalignment = 'center'
            self._y = 0.5
        else:
            self._verticalalignment = 'top'
            self._y = 1.0

        # Horizontal
        if position in ("sw", "w", "nw"):
            self._horizontalalignment = 'left'
            self._x = 0.0
        elif position in ("n", "s"):
            self._horizontalalignment = 'center'
            self._x = 0.5
        else:
            self._horizontalalignment = 'right'
            self._x = 1.0

        self._backgroundcolor = backgroundcolor
        self._color = color
        self._text = None

    def set_text(self, s):
        self.text.set_text(s=s)

    def initialize(self):
        self._text = plt.text(
            x=self._x,
            y=self._y,
            s="",
            horizontalalignment=self._horizontalalignment,
            verticalalignment=self._verticalalignment,
            transform=plt.gca().transAxes,
            backgroundcolor=self._backgroundcolor,
            color=self._color
        )

    @property
    def text(self):
        return self._text


class Video:
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3,
                 time_left="ne", block=True):
        """
        :param fig:
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second
        :param int | float seconds: The length of the video.
        """
        self.frames = None  # type: list
        self.artists = []
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._seconds = seconds
        self._n_frames = int(frame_rate * seconds)
        self._frame_size = None
        self._video_is_over = False
        self._frame_nr = None

        # For showing time left of video
        self._time_left = None
        self._time_left_text = None  # type: VideoTexter
        self._show_time_left = time_left
        self._start_time = None

        # Test getting frame
        self._current_frame = get_photo()
        self._frame_size = self._current_frame.shape

        # Default figure
        if fig is None:
            fig = plt.figure()
        self._fig = fig
        self._canvas = self._fig.canvas

        # Get axes and remove ticks
        self._ax = plt.gca()
        self._ax.xaxis.set_ticks([])
        self._ax.yaxis.set_ticks([])

        # Store animation
        self._image_plot = None
        self._animation = animation.FuncAnimation(
            self._fig,
            self.__animate_step,
            init_func=self.__initialize_animation,
            interval=1000 / frame_rate,
            repeat=False,
            frames=self._n_frames + 10,
        )

        # Block main thread
        if block:
            self._wait_for_end()
            plt.close(self._fig)

    def _end_of_video(self):
        self._image_plot.set_data(np.ones(shape=self._frame_size))
        plt.text(
            x=self._frame_size[1] * 0.5,
            y=self._frame_size[0] * 0.5,
            s="End of video.",
            ha="center"
        )
        self._video_is_over = True

    def _initialize_animation(self):
        pass

    def _animate_step(self, i):
        pass

    def __initialize_animation(self):
        self.artists = []
        self._video_is_over = False
        if self._store_frames:
            self.frames = []

        # Get and set photo
        self._frame_nr = 0
        self._current_frame = get_photo()
        self._image_plot = plt.imshow(self._current_frame)
        plt.draw()
        plt.show()
        self.artists.append(self._image_plot)

        # Set time-left text
        if self._show_time_left is not None:
            self._time_left = self._seconds
            self._time_left_text = VideoTexter(backgroundcolor="darkgreen", position=self._show_time_left)
            self._time_left_text.initialize()
            self.artists.append(self._time_left_text.text)

        # Allow additional artists from child classes
        self._initialize_animation()

        # Set time of start
        self._start_time = time()

        return self.artists

    def __animate_step(self, i):
        self._frame_nr = i

        # Get and set photo
        self._current_frame = get_photo()
        self._image_plot.set_data(self._current_frame)

        # Update time-left text
        if self._show_time_left is not None:
            c_time_left = self._seconds - time() + self._start_time
            c_time_left = c_time_left if c_time_left > 0 else 0
            self._time_left_text.set_text("{:.1f}s".format(c_time_left))

        # Allow updating additional artists from child classes
        self._animate_step(i=i)

        # Frame storage
        if self._store_frames:
            self.frames.append(self._current_frame)

        # Check for end
        if time() >= self._start_time + self._seconds:
            self._fig.canvas.stop_event_loop()
            self._end_of_video()

        return self.artists

    def _wait_for_end(self):
        while not self._video_is_over:
            plt.pause(0.5)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = Video()
