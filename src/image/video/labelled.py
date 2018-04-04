from src.image.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.base import VideoTexter, Video
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np


class LabelledVideo(Video):
    def __init__(self, model, fig=None, record_frames=False, frame_rate=5, seconds=3, time_left="ne", block=True, title="Video",
                 backgroundcolor="darkblue", color="white"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure while labelling them with a machine learning
        model.
        :param ImageLabeller model: Model for labelling images.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float seconds: The length of the video.
        :param None | str time_left: Position of count-down timer. None if no timer is wanted.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.

        :param str backgroundcolor: Color of background of label-text.
        :param str color: Face color of label-text.
        """
        self._text = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self._model = model
        self._colors = self._make_colormap()

        super().__init__(fig=fig, record_frames=record_frames, frame_rate=frame_rate, seconds=seconds,
                         time_left=time_left, block=block, title=title)

    def _make_colormap(self):
        # Make a user-defined colormap.
        cm1 = colors.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])

        # Make a normalizer that will map the time values from
        # [start_time,end_time+1] -> [0,1].
        cnorm = colors.Normalize(vmin=0.0, vmax=1.0)

        # Turn these into an object that can be used to map time values to colors and
        # can be passed to plt.colorbar().
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        cpick.set_array(np.ndarray([]))

        return cpick

    def _initialize_animation(self):
        # Initialize video
        super()._initialize_animation()

        # Initialize text
        self._text.initialize()

        return self.artists

    def _animate_step(self, i):
        # Update video
        _ = super()._animate_step(i=i)

        # Get labels and probabilities
        labels, probabilities = self._model.label_frame(frame=self._current_frame)

        # Write some text
        self._text.set_text(labels[0])
        self._text.set_background_color(self._colors.to_rgba(probabilities[0]))

        return self.artists


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    labelling_model = KerasDetector()
    the_video = LabelledVideo(
        model=labelling_model,
        seconds=10
    )
