""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from src.image.object_detection.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector

from src.real_time.frontend_opencv import OpenCVFrontendController
from src.real_time.frontend_matplotlib import MatplotlibFrontendController
from src.image.backend_opencv import OpenCVBackendController

from time import time

# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

# pylint: disable=E0202
class LoopInterface:
    """
    Interface for a backend. 5 Functions can be set for performing tasks at various relevant times.
    The wanted loop-time can also be set.
    """
    def __init__(self, loop_initialize=noop, loop_step=noop, loop_stop_check=noop, loop_finalize=noop,
                 loop_interrupt_handler=noop, loop_time_milliseconds=200):
        """
        Dynamic version of the backend interface.
        :param Callable loop_initialization:
        :param Callable loop_step:
        :param Callable loop_stop_check:
        :param Callable finalize:
        :param Callable loop_interrupt_handler:
        :param int loop_time_milliseconds:
        """
        self._loop_initialize = loop_initialize
        self._loop_step = loop_step
        self._loop_stop_check = loop_stop_check
        self._loop_finalize = loop_finalize
        self._loop_interrupt_handler = loop_interrupt_handler
        self._loop_time_milliseconds = loop_time_milliseconds

    def _loop_initialize(self):
        pass

    @property
    def loop_initialize(self):
        return self._loop_initialize

    def _loop_step(self):
        pass

    @property
    def loop_step(self):
        return self._loop_step

    def _loop_stop_check(self):
        pass

    @property
    def loop_stop_check(self):
        return self._loop_stop_check

    def _loop_finalize(self):
        pass

    @property
    def loop_finalize(self):
        return self._loop_finalize

    def _loop_interrupt_handler(self):
        pass

    @property
    def loop_interrupt_handler(self):
        return self._loop_interrupt_handler

    def _loop_time_milliseconds(self):
        pass

    @property
    def loop_time_milliseconds(self):
        return self._loop_time_milliseconds



class VideoLoop:
    def __init__(self, model, store_predictions=False,
                 frame_rate = 24, stream_type = "thread",
                 video_length = 3, length_is_nframes = False,
                 record_frames = False,
                 title = "Video", 
                 verbose = False,
                 show_crosshair = True, show_labels = True,
                 cutout_size = (224, 224),
                 backend = "opencv", frontend = "opencv", video_path = None):
        # Video settings
        self.frame_rate = frame_rate
        self.stream_type = stream_type
        self.record_frames = record_frames
        self.frame_rate = frame_rate
        self.frame_time = int(1000 * 1. / frame_rate)
        self.video_length = video_length
        self.length_is_nframes = length_is_nframes
        self.title = title
        self.verbose = verbose
        self.show_crosshair = show_crosshair
        self.show_labels = show_labels

        # Model settings
        self.cutout_size = cutout_size
        self.predictions = None  # type: list
        self.cut_frames = []
        self.current_label = None
        self.current_label_probability = 0.0
        self.store_predictions = store_predictions
        self.model = model

        # # Set frame size
        # if not video_path == None:
        #     dummy_video = VCR(video_path) 
        #     self._frame_size = dummy_video.get_frame_size_only()

        # Getting the frontend interface
        interface = LoopInterface(
            loop_initialize=self.loop_initialize,
            loop_step=self.loop_step,
            loop_stop_check=self.loop_stop_check,
            loop_interrupt_handler = self.loop_interrupt_handler,
            loop_finalize=self.loop_finalize,
            loop_time_milliseconds=self.frame_time
        )
        
        # Getting the backend object
        if backend == "opencv":
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate)
        else:
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate)

        # Getting the frontend object
        if frontend == "opencv":
            self.frontend = OpenCVFrontendController(interface = interface, 
                                                     title = self.title,
                                                     show_crosshair = self.show_crosshair, 
                                                     show_labels = self.show_labels)
        if frontend == "matplotlib":
            self.frontend = MatplotlibFrontendController(interface = interface, 
                                                     title = self.title,
                                                     show_crosshair = self.show_crosshair, 
                                                     show_labels = self.show_labels)
        else:
            self.frontend = OpenCVFrontendController(interface = interface, 
                                                     title = self.title,
                                                     show_crosshair = self.show_crosshair, 
                                                     show_labels = self.show_labels)
        
    def start(self):
        # Run loop on the frontend
        self.frontend.run()

    def loop_initialize(self):
        self.current_frame = self.backend.current_frame
        self.current_frame_time = time()
        self.frame_size = self.current_frame.shape
        # Forwarding currrent parameters size to frontend
        self.frontend.frame_size = self.frame_size
        self.frontend.title = self.title
        
    def loop_step(self):
        
        # Get the frame
        self.current_frame = self.backend.current_frame
        self.current_frame_time = time()

        # Get first frame cutout
        self.current_frame_cutout = self.backend.current_frame_cutout(frame = self.current_frame, frame_size = self.frame_size)
        
        # Get labels and probabilities for the frame cutout
        labels, probabilities = self.model.label_frame(frame=self.current_frame_cutout)

        # Set analysis
        self.current_label = labels[0]
        self.current_label_probability = probabilities[0]

        # Forward current frame parameters to the frontend
        self.frontend.current_label = self.current_label
        self.frontend.current_label_probability = self.current_label_probability
        self.frontend.current_frame = self.current_frame

        # Store cut frames if stated
        if self.record_frames:
            self.cut_frames.append(self.current_frame_cutout)
        
         # Storage
        if self.store_predictions:
            self.predictions.append((
                self.current_frame_time, labels, probabilities
            ))

    def loop_finalize(self):
        pass

    def loop_interrupt_handler(self):
        pass

    def loop_stop_check(self):
        this_is_the_end = False

        if self.video_length is None:
            this_is_the_end = False
        elif self.length_is_nframes and self.frontend.current_loop_nr >= self.video_length - 1:
            this_is_the_end = True
        elif time() >= self.frontend.start_time + self.video_length:
            this_is_the_end = True

        if this_is_the_end:
            self.dprint("\tEnd condition met.")

        return this_is_the_end
    
    def dprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == "__main__":
    labelling_model = KerasDetector(model_specification="mobilenet")
    videoloop = VideoLoop(model = labelling_model, video_length = None, frontend = "matplotlib")
    videoloop.start()

