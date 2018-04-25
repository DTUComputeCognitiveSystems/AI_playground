import ctypes
from multiprocessing import Process, Array, Event as MEvent
from threading import Thread, Lock, Event, Timer
from time import time, sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np


class _CameraHolder:
    camera = None
    n_users = 0
    close_wait = 20.0

    def __init__(self):
        raise NotImplementedError("_CameraHolder is a singleton class-object and shall not be initialized.")

    @staticmethod
    def _get_capturer():
        """
            Gets a video/photo capturer from cv2 and returns the same object every time.
            :rtype: cv2.VideoCapture
            """
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            raise IOError("Camera could not be opened. Probably already in use.")
        _, out = cam.read()

        if np.mean(out) < 0.5:
            raise IOError(
                "Camera could not be opened (based on output). Probably already in use. "
                "Try to close other applications.")

        return cam

    @staticmethod
    def get_camera():
        if _CameraHolder.camera is None:
            _CameraHolder.camera = _CameraHolder._get_capturer()

        _CameraHolder.n_users += 1
        return _CameraHolder.camera

    @staticmethod
    def _try_closing():
        if _CameraHolder.n_users < 1 and _CameraHolder.camera is not None:
            _CameraHolder.camera.release()
            _CameraHolder.camera = None

    @staticmethod
    def release():
        _CameraHolder.n_users -= 1

        # If no one is using the camera, then consider closing it in 30 seconds
        if _CameraHolder.n_users < 1:
            timer = Timer(_CameraHolder.close_wait, _CameraHolder._try_closing)
            timer.start()


class AutoClosingCapturer:
    def __init__(self):
        self._cam = _CameraHolder.get_camera()

    def raw_read(self):
        return self._cam.read()

    def get_photo(self):
        # Read and wait
        _, out = self.raw_read()

        while out is None:
            pass

        # Reverse last dimension (CV2 and plotting libraries apparently work differently with images)
        out = out[:, :, ::-1]

        return out

    def release(self):
        if self._cam is not None:
            self._cam = None
            _CameraHolder.release()

    def __del__(self):
        self.release()


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
        capturer = AutoClosingCapturer()
        return capturer.get_photo()

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
        cam = AutoClosingCapturer()
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    with self.lock:
                        self._current_frame = cam.get_photo()
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
        cam = AutoClosingCapturer()
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    c_photo = cam.get_photo()  # type: np.ndarray
                    self._current_frame[:] = c_photo[:]

        except KeyboardInterrupt:
            pass


class CameraStreamProcess(_CameraStreamer):
    def __init__(self, frame_rate=5):
        temp = AutoClosingCapturer()
        test_frame = temp.get_photo()
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
    plt.ion()

    a_photo = AutoClosingCapturer().get_photo()
    plt.imshow(a_photo)
    plt.draw()
    plt.show()

    for i in range(int(_CameraHolder.close_wait + 5)):
        print("{}: {}".format(i, _CameraHolder.camera))
        sleep(1)
