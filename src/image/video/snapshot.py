from src.image.video.base import Video, VideoTexter
import matplotlib.pyplot as plt
from datetime import datetime


class VideoCamera(Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, video_length=3,
                 length_is_nframes=False, time_left="ne", block=True, title="Camera",
                 backgroundcolor="darkblue", color="white"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float video_length: The length of the video.
        :param None | str time_left: Position of count-down timer. None if no timer is wanted.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.

        :param str backgroundcolor: Color of background of camera-text.
        :param str color: Face color of camera-text.
        """
        self._texter = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self.photos = []
        self.photos_info = []
        self.length_is_nframes = length_is_nframes


        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, video_length=video_length,
                         length_is_nframes=length_is_nframes, time_left=time_left, block=block, title=title)

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
    the_video = VideoCamera(video_length=10)
