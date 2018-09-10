""" temporary - lines 1-8"""
import git
import os
import sys
git_root = git.Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel") # '.' causes issue on windows/osx?
os.chdir(git_root)
sys.path.insert(0, git_root)

import random
from time import time

from src.image.object_detection.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.base import _Video, MATPLOTLIB_BASED_BACKENDS, OPENCV_BASED_BACKENDS
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import cv2

from src.image.video.texter import VideoTexter
from src.real_time.background_backend import BackgroundLoop
from src.real_time.matplotlib_backend import MatplotlibLoop
from src.real_time.opencv_backend import OpenCVLoop
from src.image.video.snapshot import CrossHair, FrameCutout

class OpenCVVideoEffects():
    def __init__(self, opencv_frame = None, opencv_text = "", crosshair = {}, frame_size = None, crosshair_size = None):
        
        """
        Adds the overlay effects to the video: captions for the detected objects and crosshair.
        :param opencv_frame: current webcamera frame from the stream
        :param str opencv_text: text to put on the frame (object label)
        :param dict crosshair: crosshair parameters [color, thickness]
        :param tuple frame_size: size of the frame
        :param tuple crosshair_size: size of the crosshair
        """
        
        self.opencv_frame = opencv_frame
        self.opencv_text = opencv_text
        self.crosshair = crosshair
        self.frame_size = frame_size
        self.crosshair_size = crosshair_size

    def update(self):
        # Calculating the top-left and bottom-right angles of the rectangle
        crosshair_point_1 = (int(self.frame_size[1] / 2 - self.crosshair_size[1] / 2), int(self.frame_size[0] / 2 - self.crosshair_size[0] / 2))
        crosshair_point_2 = (int(self.frame_size[1] / 2 + self.crosshair_size[1] / 2), int(self.frame_size[0] / 2 + self.crosshair_size[0] / 2))
        # Getting the text size on the screen to align it nicely
        text_size, _ = cv2.getTextSize(self.opencv_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        # Calculating the coordinates to put text on the screen
        text_bg_point_1 = (self.frame_size[1] - text_size[0] - 10,0)
        text_bg_point_2 = (self.frame_size[1], text_size[1] + 15)
        # Drawing the text background
        print(sys.getsizeof(self.opencv_frame))
        cv2.rectangle(self.opencv_frame, text_bg_point_1, text_bg_point_2, (255,255,255), -1)
        # Displaying the text on the frame
        cv2.putText(self.opencv_frame, self.opencv_text,(self.frame_size[1] - text_size[0] - 5, text_size[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
        # Drawing the crosshair
        cv2.rectangle(self.opencv_frame, crosshair_point_1, crosshair_point_2, self.crosshair["color"], self.crosshair["thickness"], cv2.LINE_8)

    def setText(self, opencv_text):
        self.opencv_text = opencv_text

    def setFrame(self, opencv_frame):
        self.opencv_frame = opencv_frame

    def setCrossHair(self):
        self.crosshair["color"] = (0, 0, 255) #red
        self.crosshair["thickness"] = 3
        

class LabelledVideo(_Video):
    def __init__(self,
                 model, backgroundcolor="darkblue", color="white", store_predictions=False,
                 frame_rate=5, stream_type="thread",
                 video_length=3, length_is_nframes=False,
                 record_frames=False,
                 title="Video", ax=None, fig=None, opencv_frame=None, block=True,
                 verbose=False, print_step=1,
                 crosshair_type='box',
                 crosshair_size=(224, 224),
                 backend="matplotlib",video_path = None):
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
            blit=False,
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
            self.opencveffects.setText(labels[0])
            self.opencveffects.setCrossHair()
            self.opencveffects.update()

if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    backends = [MatplotlibLoop(block=True), BackgroundLoop(), OpenCVLoop(title="Webcam video stream")]
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

