from time import sleep

import cv2
import matplotlib
from matplotlib import image
import numpy as np

matplotlib.use('TkAgg')
from matplotlib import animation
import matplotlib.pyplot as plt
from functools import lru_cache


@lru_cache(maxsize=1)
def get_capturer():
    """
    Gets a video/photo capturer from cv2 and returns the same object every time.
    :rtype: cv2.VideoCapture
    """
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise IOError("Camera could not be opened. Probably already in use.")

    return cam


def get_photo(photo_capturer=None):
    """
    Fetches a photo from webcam.
    :param cv2.VideoCapture photo_capturer:
    :rtype: np.ndarray
    """
    if photo_capturer is None:
        photo_capturer = get_capturer()

    # Read and wait
    _, out = photo_capturer.read()
    while out is None:
        pass

    return out


class Video:
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3, block=True):
        """
        :param fig:
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second
        :param int | float seconds: The length of the video.
        """
        self.frames = None  # type: list
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._seconds = seconds
        self._n_frames = int(frame_rate * seconds)
        self._frame_size = None
        self._video_is_over = False

        # Test getting frame
        self._current_frame = get_photo()
        self._frame_size = self._current_frame.shape

        # Default figure
        if fig is None:
            fig = plt.figure()
        self._fig = fig

        # Get axes and remove ticks
        self._ax = plt.gca()
        self._ax.xaxis.set_ticks([])
        self._ax.yaxis.set_ticks([])

        # Store animation
        self._image_plot = None
        self._animation = animation.FuncAnimation(
            self._fig,
            self._animate_step,
            init_func=self._initialize_animation,
            interval=1000/frame_rate,
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
        self._video_is_over = False
        if self._store_frames:
            self.frames = []

        # Get and set photo
        self._current_frame = get_photo()
        self._image_plot = plt.imshow(self._current_frame)  # type: image.AxesImage
        plt.draw()
        plt.show()

        return self._image_plot,

    def _animate_step(self, i):
        # Get and set photo
        self._current_frame = get_photo()
        self._image_plot.set_data(self._current_frame)

        # Frame storage
        if self._store_frames:
            self.frames.append(self._current_frame)

        if i >= self._n_frames:
            self._fig.canvas.stop_event_loop()
            self._end_of_video()

        return self._image_plot,

    def _wait_for_end(self):
        while not self._video_is_over:
            plt.pause(0.5)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = Video()
