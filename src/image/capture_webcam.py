import cv2
import numpy as np
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
