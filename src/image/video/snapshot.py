from datetime import datetime

import matplotlib.pyplot as plt

from src.image.video.base import _Video
from src.image.video.texter import VideoTexter


class VideoCamera(_Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, n_photos=5,
                 block=True, title="Camera", stream_type="simple",
                 backgroundcolor="darkblue", color="white", ax=None, backend="matplotlib"):
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
            backend=backend
        )

        self._texter = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self.flairs.extend([self._texter])
        self.photos = []
        self.photos_info = []
        self.n_photos = n_photos

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

    def _animate_video_extensions(self):
        # Write some text
        self._texter.set_text(self._camera_text())


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoCamera(n_photos=5, stream_type="process")
    the_video.start()

    print("Number of picutres taken: {}".format(len(the_video.photos)))
