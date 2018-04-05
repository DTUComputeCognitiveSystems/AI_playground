import cv2
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt


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

    # Reverse last dimension (CV2 and plotting libraries apparently work differently with images)
    out = out[:, :, ::-1]

    return out


if __name__ == "__main__":

    a_photo = get_photo()
    plt.imshow(a_photo)
