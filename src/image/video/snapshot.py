from src.image.video.base import Video, VideoTexter
import matplotlib.pyplot as plt
from datetime import datetime


class VideoCamera(Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3, block=True,
                 backgroundcolor="darkblue", color="white", title="Camera"):
        self._texter = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self.photos = []
        self.photos_info = []

        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, seconds=seconds,
                         block=block, title=title)

    def _initialize_animation(self):
        # Initialize video
        super()._initialize_animation()

        # Set snapshot event
        self._canvas.mpl_connect("key_press_event", self._take_photo)

        # Initialize texter
        self._texter.initialize()
        self.artists.append(self._texter.text)

    def _camera_text(self):
        return "Camera\nPhotos taken: {}".format(len(self.photos))

    def _animate_step(self, i):
        # Update video
        _ = super()._animate_step(i=i)

        # Write some text
        self._texter.set_text(self._camera_text())

    def _take_photo(self, event):
        key = event.key

        if "enter" in key:
            c_time = datetime.now()
            self.photos.append(self._current_frame)
            self.photos_info.append((self._frame_nr, str(c_time.date()), str(c_time.time())))


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoCamera(seconds=10)
