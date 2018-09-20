import ctypes
from multiprocessing import Process, Array, Event as MEvent
from threading import Thread, Lock, Event, Timer
from time import time, sleep
from os.path import isfile

import cv2
import numpy as np


class Camera:
    camera_access = None
    timer = Timer(1, lambda: print(""))
    close_wait = 5.0

    def __init__(self):
        raise NotImplementedError("_Camera is a singleton class-object and should not be initialized.")

    @staticmethod
    def _get_capturer():
        """
            Gets a video/photo capturer from cv2 and returns the same object every time.
            :rtype: cv2.VideoCapture
            """
        cam = cv2.VideoCapture(0)

        # Check access
        if not cam.isOpened():
            raise IOError("Camera could not be opened. Probably already in use.")

        # Test output
        _, out = cam.read()
        if np.mean(out) < 0.5:
            raise IOError(
                "Camera could not be opened (based on output). Probably already in use. "
                "Try to close other applications.")

        return cam

    @staticmethod
    def close():
        if Camera.camera_access is not None:
            Camera.camera_access.release()
            Camera.camera_access = None

    @staticmethod
    def _ensure_camera():
        if Camera.camera_access is None:
            Camera.camera_access = Camera._get_capturer()

    @staticmethod
    def raw_read():
        # Stop timer
        Camera.timer.cancel()

        # Make sure camera is accessed and read
        Camera._ensure_camera()
        output = Camera.camera_access.read()

        # Reset timer
        Camera.timer = Timer(Camera.close_wait, Camera.close)
        Camera.timer.start()

        return output

    @staticmethod
    def get_photo():
        # Read and wait
        _, out = Camera.raw_read()

        while out is None:
            pass

        # Reverse last dimension (CV2 apparently loads images in BGR format)
        out = out[:, :, ::-1]
        out = cv2.flip(out, 1)

        return out


class _CameraStreamer:
    @property
    def current_frame(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class SimpleStream(_CameraStreamer):
    @property
    def current_frame(self):

        return Camera.get_photo()

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
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    with self.lock:
                        self._current_frame = Camera.get_photo()
                        self._frame_count += 1

        except KeyboardInterrupt:
            pass

        Camera.close()


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
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    c_photo = Camera.get_photo()  # type: np.ndarray
                    self._current_frame[:] = c_photo[:]

        except KeyboardInterrupt:
            pass

        Camera.close()


class CameraStreamProcess(_CameraStreamer):
    def __init__(self, frame_rate=5):
        test_frame = Camera.get_photo()
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


class VCR:
    camera_access = None

    close_wait = 5.0

    def __init__(self, file_source):
        """ can be used to iterate over an iterable or open a video"""
        self.is_iterable = False
        if type(file_source) == str:

            if not isfile(file_source):
                raise OSError("File not found or is not a file. Please check path")
            self.file_source = file_source
            self.cam = cv2.VideoCapture(self.file_source)
            if not self.cam.isOpened():
                raise IOError("Camera could not be opened. Probably already in use.")
        else:
            # assume n iterator was given

            self.cam = iter(file_source)
            self.is_iterable = True

    def get_frame_size_only(self):
        # if this is a python object, we just assume there is enough frames that it doesn't really matter
        if self.is_iterable:
            shape = next(self.cam).shape
        else:
            shape = self.get_frame()[1].shape
            self.close()
        return shape

    def close(self):
        if not self.is_iterable:
            self.cam.release()

    def get_frame(self):
        # Read and wait
        if self.is_iterable:
            try:
                frame = next(self.cam)
                return True, frame
            except StopIteration:
                return False, None
        else:
            is_valid, out = self.cam.read()
            if is_valid:
                # Reverse last dimension (CV2 apparently loads images in BGR format)
                out = out[:, :, ::-1]
                out = cv2.flip(out, 1)
                return is_valid, out
            else:
                return False, None


class VCRStream(Thread):
    def __init__(self, video_path, frame_rate=5):
        super().__init__()
        self.frame_rate = frame_rate
        self._frame_time = 1. / frame_rate
        self.daemon = True
        self.lock = Lock()
        self._current_frame = None
        self._frame_count = 0
        self.vcr = VCR(video_path)

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
            sleep(0.04)
            with self.lock:
                current_frame = self._current_frame

        return current_frame

    @property
    def frame_count(self):
        with self.lock:
            count = self._frame_count
        return count

    def run(self):
        next_time = time()

        try:
            while True:
                if self._stop_event.is_set():
                    break

                c_time = time()
                if c_time > next_time:
                    next_time = c_time + self._frame_time
                    with self.lock:
                        is_valid, self._current_frame = self.vcr.get_frame()
                        self._frame_count += 1
                    if not is_valid:
                        break

        except KeyboardInterrupt:
            pass

        self.vcr.close()


class OpenCVBackendController():
    def __init__(self, frame_rate, stream_type = "stream", video_path = None):
        self.frame_rate = frame_rate
        self.video_path = video_path
        self.stream_type = stream_type
        self.camera_stream = None

        # Open up a camera-stram
        if self.video_path == None:
            if "process" in self.stream_type:
                # Video: Multiprocessing
                self.camera_stream = CameraStreamProcess(frame_rate=self.frame_rate)
            elif "thread" in self.stream_type:
                # Video: Multithreaded.
                self.camera_stream = CameraStream(frame_rate=self.frame_rate)
            else:
                # Video: Simple
                self.camera_stream = SimpleStream()
        else:
            self.camera_stream = VCRStream(self.video_path)

        # Get photo
        self._current_frame = self.camera_stream.current_frame
        self._current_frame_time = time()

    @property
    def current_frame(self):
        self.current_frame_time = time()
        return self.camera_stream.current_frame
    
    def stop(self):
        return self.camera_stream.stop()
