import ctypes
from threading import Thread, Lock, Event
from multiprocessing import Process, Array, Event as MEvent
from time import time, sleep

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

def get_photo_on_keypress(photo_capturer=None, name='frame'):
    """
    Fetches a photo from webcam when enter is pressed
    :param cv2.VideoCapture photo_capturer
    :rtype: np.ndarray
    """
    if photo_capturer is None:
        photo_capturer = get_capturer()

    # Read and wait
    ret, frame = photo_capturer.read()
    
    while(ret):
        ret, frame = photo_capturer.read()
        cv2.imshow(name, frame)
        k = cv2.waitKey(33)
        if k != -1:
            break
    cv2.destroyAllWindows()
    
    # Reverse last dimension (CV2 and plotting libraries apparently work differently with images)
    out = frame[:, :, ::-1]
    return out


class _CameraStreamer:
    @property
    def current_frame(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class SimpleStream(_CameraStreamer):
    def __init__(self):
        pass

    @property
    def current_frame(self):
        return get_photo()

    def stop(self):
        pass


class CameraStream(Thread, _CameraStreamer):
    def __init__(self, frame_rate=5):
        super().__init__()
        self.frame_rate = frame_rate
        self._frame_time = 1. / frame_rate
        self.daemon = True
        self.lock = Lock()
        self._current_frame = None
        self._frame_count = 0

        self._stop_event = Event()

        self.start()

    def stop(self):
        self._stop_event.set()

    @property
    def current_frame(self):
        # Get frame
        with self.lock:
            current_frame = self._current_frame

        # Ensure frame
        while current_frame is None:
            sleep(0.05)
            with self.lock:
                current_frame = self._current_frame
                
        return current_frame

    @property
    def frame_count(self):
        with self.lock:
            count = self._frame_count
        return count

    def run(self):
        cam = get_capturer()
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    with self.lock:
                        self._current_frame = get_photo(photo_capturer=cam)
                        self._frame_count += 1

        except KeyboardInterrupt:
            pass


class _CameraStreamProcess(Process):
    def __init__(self, frame_manager, frame_rate=5):
        super().__init__()
        self.frame_rate = frame_rate
        self._frame_time = 1. / frame_rate

        self._stop_event = MEvent()
        self._current_frame = frame_manager

        self.start()

    def stop(self):
        self._stop_event.set()

    def run(self):
        cam = get_capturer()
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    c_photo = get_photo(photo_capturer=cam)  # type: np.ndarray
                    self._current_frame[:] = c_photo[:]

        except KeyboardInterrupt:
            pass


class CameraStreamProcess(_CameraStreamer):
    def __init__(self, frame_rate=5):
        test_frame = get_photo()
        self._frame_shape = test_frame.shape

        self._current_frame_base = Array(ctypes.c_uint8, int(np.product(self._frame_shape)))
        self._current_frame_array = np.ctypeslib.as_array(self._current_frame_base.get_obj())
        self._current_frame = self._current_frame_array.reshape(*self._frame_shape)

        self._worker = _CameraStreamProcess(frame_manager=self._current_frame, frame_rate=frame_rate)

    def stop(self):
        self._worker.stop()

    @property
    def current_frame(self):
        if self._current_frame is None:
            return None

        current_frame = self._current_frame

        if not isinstance(current_frame, np.ndarray):
            return None

        return current_frame


if __name__ == "__main__":

    a_photo = get_photo()
    plt.imshow(a_photo)


