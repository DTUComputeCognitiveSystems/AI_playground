from datetime import datetime
from time import time

import matplotlib.pyplot as plt

from src.image.video.base import _Video
from src.image.video.texter import VideoTexter
from src.real_time.matplotlib_backend import MatplotlibLoop


class VideoCamera(_Video):
    def __init__(self,
                 frame_rate=5, stream_type="simple",
                 record_frames=False,
                 n_photos=5, backgroundcolor="darkblue", color="white",
                 title="Camera", ax=None, fig=None, block=True,
                 verbose=False, print_step=1,
                 backend="matplotlib"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.
        :param int n_photos: Number of photos to take before stopping.

        :param str backgroundcolor: Color of background of camera-text.
        :param str color: Face color of camera-text.
        """
        super().__init__(
            frame_rate=frame_rate,
            video_length=None,
            length_is_nframes=False,
            record_frames=record_frames,
            title=title,
            stream_type=stream_type,
            ax=ax,
            fig=fig,
            block=block,
            blit=False,
            backend=backend,
            verbose=verbose,
            print_step=print_step
        )

        self._texter = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self.flairs.extend([self._texter])
        self.photos = []
        self.photos_info = []
        self.n_photos = n_photos

    def _step_print(self):
        self.dprint("\tVideo frame {:4d} at time {:8.2}s. {:2d} photos taken.".format(
            self.frame_nr,
            time() - self.real_time_backend.start_time,
            len(self.photos)
        ))

    def _camera_text(self):
        return "Camera\nPhotos taken: {}".format(len(self.photos))

    def _take_photo(self, event):
        key = event.key

        if "enter" in key:
            c_time = datetime.now()
            self.photos.append(self._current_frame)
            self.photos_info.append((self.frame_nr, str(c_time.date()), str(c_time.time())))

    ##################################################################
    # Video API

    def _initialize_video_extensions(self):
        if not isinstance(self.real_time_backend, MatplotlibLoop):
            raise ValueError("Camera can only be used with Matplotlib-backend. How would you see your photos? :P")

        if self.ax is None:
            self.ax = plt.gca()
        plt.sca(self.ax)

        # Set snapshot event
        self.real_time_backend.canvas.mpl_connect("key_press_event", self._take_photo)

        # Initialize texter
        self._texter.initialize()
        self._texter.set_text(self._camera_text())
        self.artists.append(self._texter.text)

    def _loop_stop_check(self):
        if len(self.photos) >= self.n_photos:
            return True
        else:
            return False

    def _step_video_extensions(self):
        # Write some text
        self._texter.set_text(self._camera_text())


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoCamera(n_photos=5, stream_type="process", verbose=True)
    the_video.start()

    print("Number of picutres taken: {}".format(len(the_video.photos)))
