from src.image.video.base import VideoTexter, Video
import matplotlib.pyplot as plt


class VideoWText(Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3, time_left="ne", block=True, title="Video",
                 get_text_function=None, backgroundcolor="darkblue", color="white"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure with an additional optional label.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float seconds: The length of the video.
        :param None | str time_left: Position of count-down timer. None if no timer is wanted.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.

        :param Callable get_text_function: Function for getting text for each frame.
        :param str backgroundcolor: Color of background of text.
        :param str color: Face color of text.
        """
        self._text = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self._get_text_function = self._get_text if get_text_function is None else get_text_function

        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, seconds=seconds,
                         time_left=time_left, block=block, title=title)

    def _initialize_animation(self):
        # Initialize video
        super()._initialize_animation()

        # Initialize text
        self._text.initialize()

        return self.artists

    def _get_text(self, i, frame):
        return "Frame #{} of shape {}".format(i, frame.shape)

    def _animate_step(self, i):
        # Update video
        _ = super()._animate_step(i=i)

        # Write some text
        self._text.set_text(self._get_text_function(i, self._current_frame))

        return self.artists


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoWText()
