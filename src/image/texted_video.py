from src.image.video_basic import VideoTexter, Video
import matplotlib.pyplot as plt


class VideoWText(Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3, block=True,
                 get_text_function=None, backgroundcolor="darkblue", color="white"):
        self._text = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self._get_text_function = self._get_text if get_text_function is None else get_text_function

        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, seconds=seconds, block=block)

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
