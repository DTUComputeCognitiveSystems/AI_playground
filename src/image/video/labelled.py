from src.image.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.base import _Video
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np

from src.image.video.texter import VideoTexter


class LabelledVideo(_Video):
    def __init__(self, model, fig=None, record_frames=False, frame_rate=5, video_length=3,
                 length_is_nframes=False, block=True, title="Video",
                 backgroundcolor="darkblue", color="white", stream_type="process", ax=None, backend="matplotlib"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure while labelling them with a machine learning
        model.
        :param ImageLabeller model: Model for labelling images.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param int | float video_length: The length of the video.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str title: Title of video figure and canvas.
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.

        :param str backgroundcolor: Color of background of label-text.
        :param str color: Face color of label-text.
        """
        super().__init__(
            frame_rate=frame_rate,
            video_length=video_length,
            length_is_nframes=length_is_nframes,
            record_frames=record_frames,
            title=title,
            stream_type=stream_type,
            ax=ax,
            fig=fig,
            block=block,
            blit=False,
            backend=backend
        )

        # Model
        self._model = model
        self._colors = self._make_colormap()

        # Flair
        self._text = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self.flairs.extend([self._text])

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

    def _animate_video_extensions(self):
        # Update video
        _ = super()._animate_video_extensions()

        # Get labels and probabilities
        labels, probabilities = self._model.label_frame(frame=self._current_frame)

        # Write some text
        self._text.set_text(labels[0])
        self._text.set_background_color(self._colors.to_rgba(probabilities[0]))


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    labelling_model = KerasDetector(model_name="mobilenet")
    the_video = LabelledVideo(
        model=labelling_model,
        video_length=20,
    )
    the_video.start()
