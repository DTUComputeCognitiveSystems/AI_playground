from datetime import datetime
from time import time

import matplotlib.pyplot as plt
from matplotlib import patches

from src.image.video.base import _Video, VideoFlair
from src.image.video.texter import VideoTexter
from src.real_time.matplotlib_backend import MatplotlibLoop


class CrossHair(VideoFlair):
    def __init__(self, frame_size, ch_type="box", edgecolor="r", width_ratio=0.5, height_ratio=None, linewidth=1):
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
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.frame_size = frame_size
        self.edgecolor = edgecolor

    def initialize(self):
        if self.ch_type is not None:

            ax = plt.gca()

            # Default ratios
            if self.width_ratio is None and self.height_ratio is not None:
                self.width_ratio = self.height_ratio
            if self.height_ratio is None and self.width_ratio is not None:
                self.height_ratio = self.width_ratio
            assert isinstance(self.width_ratio, float) and 0 < self.width_ratio <= 1
            assert isinstance(self.height_ratio, float) and 0 < self.height_ratio <= 1

            # Determine coordinates
            width = int(self.frame_size[1] * self.width_ratio)
            w_space = self.frame_size[1] - width
            height = int(self.frame_size[0] * self.height_ratio)
            h_space = self.frame_size[0] - height

            # Make cross-hair
            if "box" in self.ch_type:
                ch = patches.Rectangle(
                    xy=(int(w_space / 2), int(h_space / 2)),
                    width=width,
                    height=height,
                    fill=False,
                    edgecolor=self.edgecolor,
                    linewidth=self.linewidth,
                )
            elif "circ" in self.ch_type:
                ch = patches.Circle(
                    xy=(int(self.frame_size[1] / 2), int(self.frame_size[0] / 2)),
                    radius=int(min(width, height) / 2),
                    fill=False,
                    edgecolor=self.edgecolor,
                    linewidth=self.linewidth,
                )
            elif "ellip" in self.ch_type:
                ch = patches.Ellipse(
                    xy=(int(self.frame_size[1] / 2), int(self.frame_size[0] / 2)),
                    width=width,
                    height=height,
                    fill=False,
                    edgecolor=self.edgecolor,
                    linewidth=self.linewidth,
                )
            else:
                raise ValueError("Unknown cross-hair type: {}.".format(self.ch_type))

            # Add to axes
            ax.add_patch(
                ch
            )

            # Append to artists
            self.artists.append(ch)

    def update(self, video):
        pass


class VideoCamera(_Video):
    def __init__(self,
                 frame_rate=5, stream_type="simple",
                 record_frames=False,
                 n_photos=5, backgroundcolor="darkblue", color="white",
                 crosshair_type="box",
                 title="Camera", ax=None, fig=None, block=True,
                 verbose=False, print_step=1,
                 backend="matplotlib"):
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
            block=block,
            blit=False,
            backend=backend,
            verbose=verbose,
            print_step=print_step
        )

        self._texter = VideoTexter(backgroundcolor=backgroundcolor, color=color)
        self._cross_hair = CrossHair(frame_size=self.frame_size, ch_type=crosshair_type)
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
    the_video = VideoCamera(n_photos=5, stream_type="process", verbose=True)
    the_video.start()

    print("Number of picutres taken: {}".format(len(the_video.photos)))
