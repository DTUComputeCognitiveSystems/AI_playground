import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

import numpy as np
from imageio import imsave
from matplotlib import pyplot as plt

from src.image.capture_webcam import CameraStream, CameraStreamProcess, SimpleStream, Camera
from src.real_time.background_backend import BackgroundLoop
from src.real_time.base_backend import BackendInterface
from src.real_time.base_backend import BackendLoop
from src.real_time.ipython_backend import IPythonLoop
from src.real_time.matplotlib_backend import MatplotlibLoop

MATPLOTLIB_BASED_BACKENDS = (MatplotlibLoop, IPythonLoop)


class VideoFlair:
    def __init__(self, artists):
        """
        :param list artists:
        """
        self._artists = artists

    def initialize(self, fig=None):
        raise NotImplementedError

    def update(self, video):
        raise NotImplementedError

    @property
    def artists(self):
        return self._artists


class _Video:
    def __init__(self, frame_rate, stream_type,
                 video_length, length_is_nframes,
                 record_frames,
                 title, ax, fig, block, blit,
                 verbose, print_step,
                 backend):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        :param int frame_rate: The number of frames per second.
        :param int | float | None video_length: The length of the video.
                                                None runs video indefinitely (or until stop-condition).
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.
        :param bool record_frames: Whether to store all the frames in a list.
        :param str title: Title of video figure and canvas.
        :param stream_type:
        :param str | BackendLoop backend: Backend to be used. If string is given this class will initialise the
                                          selected backend, otherwise the video will be interfaced to the
                                          existing backend.
        :param plt.Axes ax: Some axes to be used for the video.

        For backend="matplotlib":
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool block: Whether to wait for video to finish (recommended).
        :param blit:
        """
        # Settings
        self.frame_rate = frame_rate
        self.stream_type = stream_type
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._frame_time = int(1000 * 1. / frame_rate)
        self._title = title
        self.verbose = verbose
        self._print_step = print_step

        # Make real-time interface
        interface = BackendInterface(
            loop_initialization=self._loop_initialization,
            loop_step=self._loop_step,
            loop_stop_check=self._loop_stop_check,
            finalize=self._finalize,
            interrupt_handler=self._interrupt_handler,
            loop_time_milliseconds=self._frame_time
        )

        # Set backend
        if isinstance(backend, BackendLoop):
            self.real_time_backend = backend
            self.real_time_backend.add_interface(interface=interface)
        elif "matplotlib" in backend:
            self.real_time_backend = MatplotlibLoop(
                backend_interface=interface,
                title=title,
                fig=fig,
                block=block,
                blit=blit
            )
        elif "background" in backend:
            self.real_time_backend = BackgroundLoop(
                backend_interface=interface,
                block=block
            )
        else:
            raise NotImplementedError("Unknown backend for video. ")

        # Data-storage
        self.video_frames = []
        self.frame_times = []

        # For additional video-flair
        self.flairs = []
        self.ax = ax

        # Length of video if required
        self._length_is_frames = length_is_nframes
        self._video_length = video_length
        if video_length is None:
            self._n_frames = None
        else:
            self._n_frames = (video_length if length_is_nframes else int(frame_rate * video_length)) + 1

        # Set fields
        self._current_frame_time = None
        self.camera_stream = None

        # Set frame size
        self._current_frame = Camera.get_photo()
        self._frame_size = self._current_frame.shape

    def _initialize_video_extensions(self):
        pass

    def _step_video_extensions(self):
        pass

    def add_flair(self, texter):
        self.flairs.append(texter)

    @property
    def frame_nr(self):
        return self.real_time_backend.current_loop_nr

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def avg_frame_rate(self):
        frame_times = np.array(self.frame_times)
        avg_frame_time = (frame_times[1:] - frame_times[:-1]).mean()
        return 1 / avg_frame_time

    def save_recording_to_video(self, destination, temp_im_format="jpeg", print_step=10,
                                pix_fmt="yuv420p"):
        if len(self.video_frames) < 1:
            raise ValueError("No video was recorded.")

        # Resolve path
        destination = Path(destination).resolve()
        self.dprint("Saving video to: {}".format(destination))

        # Delete video if it exists
        if destination.is_file():
            destination.unlink()

        # Make temporary directory for frame-images
        with TemporaryDirectory() as tempdir:
            self.dprint("Temporary directory: {}".format(tempdir))

            # Go through images
            for i, frame in enumerate(self.video_frames):
                # Print
                if (i + 1) % print_step == 0 or i == 0:
                    self.dprint("\tFrame {} / {}".format(i + 1, len(self.video_frames)))

                # Write image as .png in temporary directory
                f_path = Path(tempdir, "frame_{:010d}.".format(i) + temp_im_format)
                imsave(str(f_path), frame)

            # Command for FFMPEG for turning images into video
            self.dprint("Combining into video.")
            command = [
                '/home/jepno/anaconda3/bin/ffmpeg',
                '-r', str(int(round(self.avg_frame_rate))),
                '-i', str(Path(tempdir, 'frame_%010d.' + temp_im_format)),
                "-pix_fmt", pix_fmt,
                str(destination),
            ]

            # Make video from images
            self.dprint(" ".join(command))
            subprocess.call(
                " ".join(command),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.dprint("Done!")

    ##################################################################
    # Real-time API

    def dprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def start(self):
        self.real_time_backend.start()

    @property
    def artists(self):
        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):
            return self.real_time_backend.artists
        return None

    def _loop_initialization(self):
        # Open up a camera-stram
        if "process" in self.stream_type:
            self.camera_stream = CameraStreamProcess(frame_rate=self.frame_rate)
            if self.verbose:
                print("Video: Multiprocessing.")
        elif "thread" in self.stream_type:
            self.camera_stream = CameraStream(frame_rate=self.frame_rate)
            if self.verbose:
                print("Video: Multithreaded.")
        else:
            self.camera_stream = SimpleStream()
            if self.verbose:
                print("Video: Simple.")

        # For storage
        self.frame_times = []
        if self._store_frames:
            self.video_frames = []

        # Get photo
        self._current_frame = self.camera_stream.current_frame
        self._current_frame_time = time()

        # Printing
        self.dprint("Initializing video.")

        # Plotting if using Matplotlib backend
        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):

            # Get and set axes
            self.ax = plt.gca() if self.ax is None else self.ax
            plt.sca(self.ax)

            # Add thread-stopper to closing event
            def closer(_):
                self.camera_stream.stop()

            self.real_time_backend.canvas.mpl_connect('close_event', closer)

            # Title and axis settings
            self.ax.set_title(self._title)
            if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):
                self.ax.xaxis.set_ticks([])
                self.ax.yaxis.set_ticks([])

            # Make image plot
            self._image_plot = plt.imshow(self._current_frame)
            self.artists.append(self._image_plot)
            plt.draw()
            plt.show()

            # Initialize flairs
            for flair in self.flairs:  # type: VideoFlair
                flair.initialize()
                self.artists.extend(flair.artists)

        # Allow additional artists from child classes
        self._initialize_video_extensions()

    def _step_print(self):
        self.dprint("\tVideo frame {:4d} at time {:8.2}s.".format(
            self.frame_nr, time() - self.real_time_backend.start_time))

    def _loop_step(self):
        # Get photo
        self._current_frame = self.camera_stream.current_frame
        self._current_frame_time = time()

        # Plotting if using Matplotlib backend
        if isinstance(self.real_time_backend, MATPLOTLIB_BASED_BACKENDS):

            # Update image plot
            self._image_plot.set_data(self._current_frame)

            # Update flairs
            for flair in self.flairs:  # type: VideoFlair
                flair.update(self)

        # Frame storage
        self.frame_times.append(time())
        if self._store_frames:
            self.video_frames.append(self._current_frame)

        # Allow updating additional artists from child classes
        self._step_video_extensions()

        # Printing
        if not self.frame_nr % self._print_step:
            self._step_print()

    def _loop_stop_check(self):
        this_is_the_end = False

        if self._video_length is None:
            this_is_the_end = False
        elif self._length_is_frames and self.real_time_backend.current_loop_nr >= self._video_length - 1:
            this_is_the_end = True
        elif time() >= self.real_time_backend.start_time + self._video_length:
            this_is_the_end = True

        if this_is_the_end:
            self.dprint("\tEnd condition met.")

        return this_is_the_end

    def _finalize(self):
        self.dprint("Finalizing video.")
        self.camera_stream.stop()

    def _interrupt_handler(self):
        self.dprint("Video interrupted.")
        self.camera_stream.stop()
        self.real_time_backend.stop_now = True


class SimpleVideo(_Video):
    def __init__(self,
                 frame_rate=5, stream_type="simple",
                 video_length=10, length_is_nframes=False,
                 record_frames=False,
                 title="Video", ax=None, fig=None, block=True, blit=False,
                 verbose=False, print_step=1,
                 backend="matplotlib"):
        """
        Shows the input of the webcam as a video in a Matplotlib figure.
        Do not extend this class. Instead extend _Video.
        :param int frame_rate: The number of frames per second.
        :param int | float | None video_length: The length of the video.
                                                None runs video indefinitely (or until stop-condition).
        :param bool length_is_nframes: Indicates whether the video-length is given as number of frames
                                       instead of seconds.
        :param bool record_frames: Whether to store all the frames in a list.
        :param str title: Title of video figure and canvas.
        :param stream_type:
        :param str | BackendLoop backend: Backend to be used. If string is given this class will initialise the
                                          selected backend, otherwise the video will be interfaced to the
                                          existing backend.
        :param plt.Axes ax: Some axes to be used for the video.

        For backend="matplotlib":
        :param fig: Matplotlib figure for video. Creates a new figure as default.
        :param bool block: Whether to wait for video to finish (recommended).
        :param blit:
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
            blit=blit,
            backend=backend,
            verbose=verbose,
            print_step=print_step
        )


if __name__ == "__main__":
    plt.close("all")
    plt.ion()

    # Run a visible video recording
    used_backend = MatplotlibLoop()
    used_backend.block = True
    SimpleVideo(
        video_length=3,
        backend=used_backend,
        title="Visible Video!"
    )
    used_backend.start()

    # Run video in background
    print("\n\nNon-visible video with prints:")
    used_backend = BackgroundLoop()
    the_video = SimpleVideo(
        video_length=2,
        record_frames=True,
        backend=used_backend,
        verbose=True,
        print_step=10,
        frame_rate=30
    )
    used_backend.start()

    # Print times for background video
    the_frame_times = np.array(the_video.frame_times)
    time_range = the_frame_times[-1] - the_frame_times[0]
    the_avg_frame_time = (the_frame_times[1:] - the_frame_times[:-1]).mean()
    print("Frames recored with non-visible video: {}".format(len(the_video.video_frames)))
    print("Average frame-time: {:.4f}s".format(the_avg_frame_time))
    print("Average frame-rate: {:.2f}".format(1 / the_avg_frame_time))

    # Save frames to video-file
    video_destination = Path("delete.mp4").resolve()
    print("Saving video to: {}".format(video_destination))
    the_video.save_recording_to_video(destination=video_destination)
