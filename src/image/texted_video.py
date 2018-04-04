from src.image.capture_webcam import Video
import matplotlib.pyplot as plt


class VideoWText(Video):
    def __init__(self, fig=None, record_frames=False, frame_rate=5, seconds=3, block=True,
                 get_text_function=None, backgroundcolor="darkblue", color="white"):
        self._text = None
        self._get_text_function = self._get_text if get_text_function is None else get_text_function
        self.backgroundcolor = backgroundcolor
        self.color = color

        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, seconds=seconds, block=block)

    def _initialize_animation(self):
        # Initialize video
        super()._initialize_animation()

        # Create text
        self._text = plt.text(
            x=0,
            y=0,
            s="",
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=self._ax.transAxes,
            backgroundcolor=self.backgroundcolor,
            color=self.color
        )

        return self._image_plot, self._text

    def _get_text(self, i, frame):
        return "Frame #{} of shape {}".format(i, frame.shape)

    def _animate_step(self, i):
        # Update video
        _ = super()._animate_step(i=i)

        # Write some text
        self._text.set_text(self._get_text_function(i, self._current_frame))

        return self._image_plot, self._text


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoWText()
