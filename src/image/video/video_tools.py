""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from datetime import datetime
from time import time

import matplotlib.pyplot as plt
from matplotlib import patches

import cv2

from src.image.video.base import _Video, VideoFlair
from src.real_time.matplotlib_backend import MatplotlibLoop


class FrameCutout:
    def __init__(self, frame_size, size=None, width_ratio=0.5, height_ratio=None):
        # Check if specific size is wanted
        if size is not None:
            height, width = size

        # Otherwise compute size from ratios
        else:
            # Ensure two ratios
            if width_ratio is None and height_ratio is not None:
                width_ratio = height_ratio
            if height_ratio is None and width_ratio is not None:
                height_ratio = width_ratio
            assert isinstance(width_ratio, float) and 0 < width_ratio <= 1
            assert isinstance(height_ratio, float) and 0 < height_ratio <= 1

            # Determine cutout-size
            width = int(frame_size[1] * width_ratio)
            height = int(frame_size[0] * height_ratio)

        # Determine space around cutout
        horizontal_space = frame_size[1] - width
        vertical_space = frame_size[0] - height

        # Determine coordinates
        coordinates = (int(horizontal_space / 2), int(vertical_space / 2), width, height)

        # Set fields
        self.coordinates = coordinates
        self.horizontal_space = horizontal_space
        self.vertical_space = vertical_space
        self.width = width
        self.height = height


class MatplotlibCrossHair(VideoFlair):
    def __init__(self, frame_size, ch_type="box", edgecolor="r", width_ratio=0.5, height_ratio=None, linewidth=1,
                 size=None):
        """

        :param tuple | list frame_size:
        :param str | None ch_type:
        :param str | tuple edgecolor:
        :param float | None width_ratio:
        :param float | None height_ratio:
        :param float | int linewidth:
        """
        super().__init__([])
        self.ch_type = ch_type
        self.linewidth = linewidth
        self.frame_size = frame_size
        self.edgecolor = edgecolor

        # Make cutout of frame
        self.frame_cutout = FrameCutout(
            frame_size=self.frame_size,
            size=size,
            width_ratio=width_ratio,
            height_ratio=height_ratio
        )

    def initialize(self, **kwargs):
        fc = self.frame_cutout

        # Make cross-hair
        if self.ch_type is None or "box" in self.ch_type:
            ch = patches.Rectangle(
                xy=(int(fc.horizontal_space / 2), int(fc.vertical_space / 2)),
                width=fc.width,
                height=fc.height,
                fill=False,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
            )
        elif "circ" in self.ch_type:
            ch = patches.Circle(
                xy=(int(self.frame_size[1] / 2), int(self.frame_size[0] / 2)),
                radius=int(min(fc.width, fc.height) / 2),
                fill=False,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
            )
        elif "ellip" in self.ch_type:
            ch = patches.Ellipse(
                xy=(int(self.frame_size[1] / 2), int(self.frame_size[0] / 2)),
                width=fc.width,
                height=fc.height,
                fill=False,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
            )
        else:
            raise ValueError("Unknown cross-hair type: {}.".format(self.ch_type))

        # Add to axes
        ax = plt.gca()
        ax.add_patch(
            ch
        )

        # Append to artists
        self.artists.append(ch)

    def update(self, video):
        pass


class MatplotlibVideoTexter(VideoFlair):
    def __init__(self, initial_text="", backgroundcolor="darkblue", color="white", position="ne"):
        super().__init__([])
        position = position.lower().strip()
        self._initial_text = initial_text

        # Vertical
        if position in ("sw", "s", "se"):
            self._verticalalignment = 'bottom'
            self._y = 0.0
        elif position in ("w", "e", "c"):
            self._verticalalignment = 'center'
            self._y = 0.5
        else:
            self._verticalalignment = 'top'
            self._y = 1.0

        # Horizontal
        if position in ("sw", "w", "nw"):
            self._horizontalalignment = 'left'
            self._x = 0.0
        elif position in ("n", "s", "c"):
            self._horizontalalignment = 'center'
            self._x = 0.5
        else:
            self._horizontalalignment = 'right'
            self._x = 1.0

        self._backgroundcolor = backgroundcolor
        self._color = color
        self._text = None  # type: matplotlib.text.Text

    def initialize(self, fig=None):
        fig = plt if fig is None else fig
        self._text = fig.text(
            x=self._x,
            y=self._y,
            s=self._initial_text,
            horizontalalignment=self._horizontalalignment,
            verticalalignment=self._verticalalignment,
            transform=plt.gca().transAxes,
            backgroundcolor=self._backgroundcolor,
            color=self._color
        )
        self._artists.append(self.text)

    def update(self, video):
        """
        :param _Video video:
        :return:
        """
        self.text.set_text("Text")

    def set_text(self, s):
        self.text.set_text(s)

    def set_background_color(self, new_color):
        self._text.set_backgroundcolor(new_color)

    def color(self, new_color):
        self._text.set_color(new_color)

    @property
    def text(self):
        return self._text


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


class VideoCamera(_Video):
    def __init__(self,
                 frame_rate=5, stream_type="simple",
                 record_frames=False,
                 n_photos=5, backgroundcolor="darkblue", color="white",
                 crosshair_type="box",
                 crosshair_size=(224, 224),
                 title="Camera", ax=None, fig=None, opencv_frame = None, block=True,
                 verbose=False, print_step=1,
                 backend="matplotlib", 
                 video_path = None):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool record_frames: Whether to store all the frames in a list.
        :param int frame_rate: The number of frames per second.
        :param bool block: Whether to wait for video to finish (recommended).
        :param str | None crosshair_type: Type of crosshair to show.
        :param str title: Title of video figure and canvas.
        :param int n_photos: Number of photos to take before stopping.

        :param str backgroundcolor: Color of background of camera-text.
        :param str color: Face color of camera-text.
        :param video_path: if this parameter is set, camera will use video instead of webcam
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
            opencv_frame = opencv_frame,
            block=block,
            blit=False,
            backend=backend,
            verbose=verbose,
            print_step=print_step,
            video_path = video_path
        )

        self._texter = MatplotlibVideoTexter(backgroundcolor=backgroundcolor, color=color)
        self._cross_hair = MatplotlibCrossHair(frame_size=self.frame_size, ch_type=crosshair_type, size=crosshair_size)
        self.cutout_coordinates = self._cross_hair.frame_cutout.coordinates
        self.flairs.extend([self._texter, self._cross_hair])
        self.photos = []
        self.photos_info = []
        self.n_photos = n_photos

    def _step_print(self):
        self.dprint("\tVideo frame {:4d} at time {:8.2}s. {:2d} photos taken.".format(
            self.frame_nr,
            time() - self.real_time_backend.start_time,
            len(self.photos)
        ))

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
        if not isinstance(self.real_time_backend, MatplotlibLoop):
            raise ValueError("Camera can only be used with Matplotlib-backend. How would you see your photos? :P")

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

    def _step_video_extensions(self):
        # Write some text
        self._texter.set_text(self._camera_text())


if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    the_video = VideoCamera(n_photos=5, stream_type="thread", verbose=True)
    the_video.start()

    print("Number of pictures taken: {}".format(len(the_video.photos)))


    # VideoTexter test
    # video = SimpleVideo(
    #     video_length=10
    # )
    # video.add_flair(VideoTexter())
    # video.start()
