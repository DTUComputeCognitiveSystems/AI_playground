""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

import random
from time import time

from src.image.object_detection.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.base import _Video, MATPLOTLIB_BASED_BACKENDS, OPENCV_BASED_BACKENDS
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import cv2

from src.image.video.videoeffects import OpenCVVideoEffects
from src.real_time.background_backend import BackgroundLoop
from src.real_time.matplotlib_backend import MatplotlibLoop
from src.real_time.opencv_backend import OpenCVLoop
from src.image.video.snapshot import CrossHair, FrameCutout, VideoTexter
        

class LabelledVideo(_Video):
    def __init__(self,
                 model, backgroundcolor="darkblue", color="white", store_predictions=False,
                 frame_rate=24, stream_type="thread",
                 video_length=3, length_is_nframes=False,
                 record_frames=False,
                 title="Video", ax=None, fig=None, opencv_frame=None, block=True, blit=False,
                 verbose=False, print_step=1,
                 crosshair_type='box',
                 crosshair_size=(224, 224),
                 backend="opencv",video_path = None):
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
            opencv_frame = opencv_frame,
            block=block,
            blit=blit,
            backend=backend,
            verbose=verbose,
            print_step=print_step,
            video_path = video_path
        )

        # Model
        self.predictions = None  # type: list
        self.cut_frames = []
        self._current_label = None
        self._current_label_probability = 0.0
        self.store_predictions = store_predictions
        self._model = model
        self._colors = self._make_colormap()

        # Default fields
        self._cross_hair = None
        self._text = None

        # If we are running matplotlib
        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):

            # Cross hair
            if crosshair_type is not None:
                self._cross_hair = CrossHair(frame_size=self.frame_size, ch_type=crosshair_type, size=crosshair_size)
                self.flairs.extend([self._cross_hair])

            # Label text
            self._text = VideoTexter(backgroundcolor=backgroundcolor, color=color)
            self.flairs.extend([self._text])

            # Cutout coordinates
            self.cutout_coord = self._cross_hair.frame_cutout.coordinates

        # If we are runnign openCV
        elif isinstance(self.real_time_backend, OPENCV_BASED_BACKENDS):
            # OpenCV video effects - cross hair and text object
            self.opencveffects = OpenCVVideoEffects()
            self.opencveffects.frame_size = self.frame_size
            self.opencveffects.crosshair_size = crosshair_size

            # Cutout
            self.cutout_coord = FrameCutout(frame_size=self.frame_size, size=crosshair_size).coordinates
        else:
            self.cutout_coord = FrameCutout(frame_size=self.frame_size, size=crosshair_size).coordinates

    def _make_colormap(self):
        cpick = None

        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):
            # Make a user-defined colormap.
            cm1 = colors.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])

            # Make a normalizer that will map the time values from
            # [start_time ,end_time+1] -> [0,1].
            cnorm = colors.Normalize(vmin=0.0, vmax=1.0)

            # Turn these into an object that can be used to map time values to colors and
            # can be passed to plt.colorbar().
            cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
            cpick.set_array(np.ndarray([]))

        return cpick

    def _step_print(self):
        self.dprint("\tVideo frame {:4d} at time {:8.2}s. Label {:^20s} at {:5.2%}.".format(
            self.frame_nr,
            time() - self.real_time_backend.start_time,
            '\'' + self._current_label + '\'',
            self._current_label_probability
        ))

    def _initialize_video_extensions(self):
        if self.store_predictions:
            self.predictions = []

    def _step_video_extensions(self):
        # Determine coordinates of cutout for grapping part of frame
        start_x, start_y, width, height = self.cutout_coord
        end_x, end_y = start_x + width, start_y + height
        frame_cut = self._current_frame[start_x:end_x, start_y:end_y]

        # Store cut frames if wated
        if self._record_frames:
            self.cut_frames.append(frame_cut)

        # Get labels and probabilities from frame-cutout
        labels, probabilities = self._model.label_frame(frame=frame_cut)

        # Set analysis
        self._current_label = labels[0]
        self._current_label_probability = probabilities[0]

        # Storage
        if self.store_predictions:
            self.predictions.append((
                self._current_frame_time, labels, probabilities
            ))

        # Visual
        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):
            # Write some text
            self._text.set_text(labels[0])
            self._text.set_background_color(self._colors.to_rgba(probabilities[0]))

        if isinstance(self.real_time_backend, OPENCV_BASED_BACKENDS):
            # Apply openCV video effects and display the text and the crosshair 
            self.opencveffects.setFrame(self.opencv_frame)
            self.opencveffects.setText("{} {:0.4f}".format(labels[0],probabilities[0]))
            self.opencveffects.setCrossHair()
            self.opencveffects.update()

        print("Frame: {}, Label:{}, Probability:{:0.4f}".format(self.frame_nr, labels[0], probabilities[0]))


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    backends = [MatplotlibLoop(block=True, blit = False), BackgroundLoop(), OpenCVLoop(title="Webcam video stream")]
    the_backend = backends[2]

    labelling_model = KerasDetector(model_specification="mobilenet")
    the_video = LabelledVideo(
        model=labelling_model,
        video_length=300,
        backend=the_backend,
        verbose=True,
        store_predictions=True,
        record_frames=True,
        stream_type="simple",
        title = the_backend.title
    )
    #the_backend.start()
    the_video.start()

    print("{} predictions made".format(len(the_video.predictions)))

    # Get a random frame number
    nr = random.randint(0, len(the_video.video_frames)-1)
    frame = the_video.video_frames[nr]
    cut_frame = the_video.cut_frames[nr]

    #plt.close("all")
    #plt.figure()
    #ax1 = plt.subplot(2, 1, 1)
    #ax1.imshow(frame)
    #ax2 = plt.subplot(2, 1, 2)
    #ax2.imshow(cut_frame)

