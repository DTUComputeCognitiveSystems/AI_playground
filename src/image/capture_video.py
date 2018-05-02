from os.path import isfile
from threading import Thread, Lock, Event
from time import time, sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == "__main__":
    plt.ion()
    test = (np.minimum(np.maximum(np.random.normal(loc=0.5, size=(224, 224, 3)), 0), 1) for i in range(200))
    my_vcr = VCR(test)

    _, a_photo = my_vcr.get_frame()
    plt.imshow(a_photo)
    plt.draw()
    plt.show()
    my_vcr.close()
