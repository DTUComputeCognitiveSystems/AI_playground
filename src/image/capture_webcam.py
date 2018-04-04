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
    def __init__(self, fig=None):

        if fig is None:
            fig = plt.figure()
        self.fig = fig

        # Initialize image plot
        frame = get_photo()
        image_plot = plt.imshow(frame)  # type: image.AxesImage
        plt.draw()
        plt.show()

        def init():
            image_plot.set_data(frame)

        def animate(i):
            frame = get_photo()
            image_plot.set_data(frame)

            return image_plot

        self.anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            interval=50
        )
        plt.show(block=True)


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = Video()
