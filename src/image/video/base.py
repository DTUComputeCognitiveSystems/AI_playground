import random

from time import time

import numpy as np
from matplotlib import pyplot as plt, animation

from src.image.capture_webcam import get_photo


class VideoFlair:
    def __init__(self, artists):
        """
        :param list artists:
        """
        self._artists = artists

    def initialize(self):
        raise NotImplementedError

    def update(self, video):
        raise NotImplementedError

    @property
    def artists(self):
        return self._artists


class Video:
    def __init__(self, frame_rate=5,
                 video_length=3, length_is_nframes=False,
                 record_frames=False,
                 title="Video",
                 fig=None, block=True):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float | None video_length: The length of the video.
                                                None runs video indefinitely (or until stop-condition).
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.
        """
        # Settings
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._title = title
        self._block = block

        # State
        self._video_is_over = False
        self._frame_nr = None
        self._start_time = None

        # Data-storage
        self.frames = []
        self._frame_times = []
        self.photos = []

        # For holding artists for Matplotlib
        self.artists = []
        self.flairs = []

        # Length of video if required
        self._length_is_frames = length_is_nframes
        self._video_length = video_length
        if video_length is None:
            self._n_frames = None
        else:
            self._n_frames = (video_length if length_is_nframes else int(frame_rate * video_length)) + 1

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

    def start(self):
        # Store animation
        self._image_plot = None
        self._animation = animation.FuncAnimation(
            self._fig,
            self.__animate_step,
            init_func=self.__initialize_animation,
            interval=1000 / self._frame_rate,
            repeat=False,
            frames=self._n_frames,
        )

        # Block main thread
        if self._block:
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

    def add_flair(self, texter):
        self.flairs.append(texter)

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

        # Initialize flairs
        for flair in self.flairs:  # type: VideoFlair
            flair.initialize()
            self.artists.extend(flair.artists)

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

        # Update flairs
        for flair in self.flairs:  # type: VideoFlair
            flair.update(self)

        # Allow updating additional artists from child classes
        self._animate_step(i=i)

        # Frame storage
        self._frame_times.append(time())
        if self._store_frames:
            self.frames.append(self._current_frame)

        # End video if needed
        if self._the_end_is_here():
                self._fig.canvas.stop_event_loop()
                self._end_of_video()

        return self.artists

    def _the_end_is_here(self):
        # Is this our inevitable doom?
        this_is_the_end = False

        if self._video_length is None:
            this_is_the_end = False
        elif self._length_is_frames and self._frame_nr >= self._video_length - 1:
            this_is_the_end = True
        elif time() >= self._start_time + self._video_length:
            this_is_the_end = True

        return this_is_the_end

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
            video_length=3,
            record_frames=True
        )
        the_video.start()
    else:
        for _ in range(10):
            n_frames = random.randint(5, 25)
            the_video = Video(
                record_frames=True,
                video_length=n_frames,
                length_is_nframes=True
            )

            print("Recored {}/{} frames".format(len(the_video.frames), n_frames))
