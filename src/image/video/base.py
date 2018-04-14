from time import time, sleep

from matplotlib import pyplot as plt

from src.image.capture_webcam import CameraStream, CameraStreamProcess, SimpleStream
from src.real_time.background_backend import BackgroundLoop
from src.real_time.base_backend import BackendInterface
from src.real_time.base_backend import BackendLoop
from src.real_time.matplotlib_backend import MatplotlibLoop


class VideoFlair:
    def __init__(self, artists):
        """
        :param list artists:
        """
        self._artists = artists

    def initialize(self):
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
        self._store_frames = record_frames
        self._frame_rate = frame_rate
        self._frame_time = int(1000 * 1. / frame_rate)
        self._title = title
        self._verbose = verbose
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

        # Open up a camera-stram
        if "process" in stream_type:
            self.camera_stream = CameraStreamProcess(frame_rate=frame_rate)
            print("Video: Multiprocessing.")
        elif "thread" in stream_type:
            self.camera_stream = CameraStream(frame_rate=frame_rate)
            print("Video: Multithreaded.")
        else:
            self.camera_stream = SimpleStream()
            print("Video: Simple.")

        # Test getting frame and get size of that frame
        self._current_frame = self._current_frame_time = None
        while self._current_frame is None:
            self._current_frame_time = time()
            self._current_frame = self.camera_stream.current_frame
            sleep(0.05)
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

    ##################################################################
    # Real-time API

    def dprint(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def start(self):
        self.real_time_backend.start()

    @property
    def artists(self):
        if isinstance(self.real_time_backend, MatplotlibLoop):
            return self.real_time_backend.artists
        return None

    def _loop_initialization(self):
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
        if isinstance(self.real_time_backend, MatplotlibLoop):

            # Get and set axes
            self.ax = plt.gca() if self.ax is None else self.ax
            plt.sca(self.ax)

            # Add thread-stopper to closing event
            def closer(_):
                self.camera_stream.stop()

            self.real_time_backend.canvas.mpl_connect('close_event', closer)

            # Title and axis settings
            self.ax.set_title(self._title)
            if isinstance(self.real_time_backend, MatplotlibLoop):
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
        self.dprint("\tideo frame {:4d} at time {:8.2}s.".format(self.frame_nr, time() - self.real_time_backend.start_time))

    def _loop_step(self):
        # Get photo
        self._current_frame = self.camera_stream.current_frame
        self._current_frame_time = time()

        # Plotting if using Matplotlib backend
        if isinstance(self.real_time_backend, MatplotlibLoop):

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
                 frame_rate=5, stream_type="process",
                 video_length=3, length_is_nframes=False,
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

    back_end = MatplotlibLoop()
    back_end.block = True

    SimpleVideo(
        video_length=3,
        backend=back_end,
        title="Visible Video!"
    )
    back_end.start()

    print("\n\nNon-visible video with prints:")
    backend = BackgroundLoop()
    video = SimpleVideo(
        video_length=3,
        record_frames=True,
        backend=backend,
        title="Visible Video!",
        verbose=True,
        print_step=2
    )
    backend.start()
    print("Frames recored with non-visible video: {}".format(len(video.video_frames)))
