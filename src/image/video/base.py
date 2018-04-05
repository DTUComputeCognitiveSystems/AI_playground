import random

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

    def set_background_color(self, new_color):
        self._text.set_backgroundcolor(new_color)

    def color(self, new_color):
        self._text.set_color(new_color)

    @property
    def text(self):
        return self._text


class Video:
    def __init__(self, fig=None, record_frames=False, frame_rate=5, video_length=3,
                 length_is_nframes=False, time_left="ne", block=True, title="Video"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float video_length: The length of the video.
        :param None | str time_left: Position of count-down timer. None if no timer is wanted.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.
        """
        self.frames = None  # type: list
        self.artists = []
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._frame_size = None
        self._video_is_over = False
        self._frame_nr = None
        self._title = title
        self._frame_times = []
        self.photos = []

        # Length of video
        self._length_is_frames = length_is_nframes
        self._video_length = video_length
        self._n_frames = video_length if length_is_nframes else int(frame_rate * video_length)
        self._video_time_length = self._video_length / self._frame_rate if self._length_is_frames \
            else self._video_length

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
        plt.title(self._title)
        self._canvas.set_window_title(self._title)

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
            frames=None if length_is_nframes else self._n_frames + 1000,
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

        if self._show_time_left is not None :
            self._time_left = self._video_time_length
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

            c_time_left = self._video_time_length - time() + self._start_time
            c_time_left = c_time_left if c_time_left > 0 else 0
            self._time_left_text.set_text("{:.1f}s".format(c_time_left))

        # Allow updating additional artists from child classes
        self._animate_step(i=i)

        # Frame storage
        self._frame_times.append(time())
        if self._store_frames:
            self.frames.append(self._current_frame)

        # Check for end
 
        video_is_done = False
        if self.length_is_nframes and len(self.photos) >= self._video_length:
            video_is_done = True


        elif not self.length_is_nframes and time() >= self._start_time + self._video_length:
            print("FF")
            video_is_done = True

        # End video if needed
        if video_is_done:
                self._fig.canvas.stop_event_loop()
                self._end_of_video()

        return self.artists

    def _wait_for_end(self):
        while not self._video_is_over:
            plt.pause(0.5)

    @property
    def frame_times(self):
        return self._frame_times


if __name__ == "__main__":
    do_frame_collecting_test = False

    plt.close("all")
    plt.ion()

    if not do_frame_collecting_test:
        the_video = Video(
            video_length=10,
        )
    else:
        for _ in range(10):
            n_frames = random.randint(5, 25)
            the_video = Video(
                record_frames=True,
                video_length=n_frames,
                length_is_nframes=True
            )

            print("Recored {}/{} frames".format(len(the_video.frames), n_frames))
