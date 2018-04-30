import ctypes
from multiprocessing import Process, Array, Event as MEvent
from threading import Thread, Lock, Event, Timer
from time import time, sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile


class VCR:
    camera_access = None
    
    close_wait = 5.0

    def __init__(self, filepath):
        if not isfile(filepath):
            raise OSError("File not found or is not a file. Please check path")
        self.filepath = filepath
        
        self.cam = cv2.VideoCapture(self.filepath)
        if not self.cam.isOpened():
            raise IOError("Camera could not be opened. Probably already in use.")
    def get_frame_size_only(self):
        shape = self.get_frame()[1].shape
        self.close()
        return shape
 



    def close(self):
        self.cam.release()


    
    def get_frame(self):
        # Read and wait
        is_valid, out = self.cam.read()
        # Reverse last dimension (CV2 apparently loads images in BGR format)
        out = out[:, :, ::-1]

        return is_valid,out




class VCRStream(Thread):
    def __init__(self,video_path, frame_rate=5):
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
    my_vcr = VCR("C:\\Users\\lauri\\Documents\\test.mp4")

    _,a_photo = my_vcr.get_frame()
    plt.imshow(a_photo)
    plt.draw()
    plt.show()
    my_vcr.close()

