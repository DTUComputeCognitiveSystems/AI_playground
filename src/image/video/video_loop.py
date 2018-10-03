""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from src.image.object_detection.keras_detector import KerasDetector

from src.real_time.interface_loop import LoopInterface
from src.real_time.frontend_opencv import OpenCVFrontendController
from src.real_time.frontend_matplotlib import MatplotlibFrontendController
from src.image.backend_opencv import OpenCVBackendController

from time import time

class VideoLoop:
    def __init__(self, regime = "object_detection", model = None, n_photos = None, store_predictions = False,
                 frame_rate = 24, stream_type = "thread",
                 video_length = None, length_is_nframes = False,
                 record_frames = False,
                 title = "Real-time video", 
                 verbose = False,
                 show_crosshair = True, show_labels = True,
                 cutout_size = (224, 224),
                 backend = "opencv", frontend = "opencv", video_path = None):
        # Video settings
        self.regime = regime #[object_detection, picture_taking, plain_camera]
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
        self.this_is_the_end = None # Determines whether the video has stopped
        self.video_path = video_path

        # Regime: object_detection - settings
        self.cutout_size = cutout_size
        self.predictions = None  # type: list
        self.cut_frames = []
        self.current_label = None
        self.current_label_probability = 0.0
        self.store_predictions = store_predictions
        self.model = model

        # Regime: picture_taking - settings
        self.n_photos = n_photos
        self.cutout_coordinates = None

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
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate, video_path = self.video_path)
        else:
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate, video_path = self.video_path)

        # Getting the frontend object
        if frontend == "opencv":
            self.frontend = OpenCVFrontendController(interface = interface, 
                                                     title = self.title,
                                                     regime = self.regime,
                                                     show_crosshair = self.show_crosshair, 
                                                     show_labels = self.show_labels)
        if frontend == "matplotlib":
            self.frontend = MatplotlibFrontendController(interface = interface, 
                                                     title = self.title,
                                                     regime = self.regime,
                                                     show_crosshair = self.show_crosshair, 
                                                     show_labels = self.show_labels)
        else:
            self.frontend = OpenCVFrontendController(interface = interface, 
                                                     title = self.title,
                                                     regime = self.regime,
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

        if self.regime == "object_detection":
            # Get first frame cutout
            self.current_frame_cutout = self.backend.current_frame_cutout(frame = self.current_frame, frame_size = self.frame_size)
            
            # Get labels and probabilities for the frame cutout
            labels, probabilities = self.model.label_frame(frame=self.current_frame_cutout)

            # Set analysis
            self.current_label = labels[0]
            self.current_label_probability = probabilities[0]

            # Forward current frame parameters to the frontend
            self.frontend.current_label = "{}: {:0.4f}".format(self.current_label, self.current_label_probability)
            
            # Store cut frames if stated
            if self.record_frames:
                self.cut_frames.append(self.current_frame_cutout)
            
            # Storage
            if self.store_predictions:
                self.predictions.append((
                    self.current_frame_time, labels, probabilities
                ))

        elif self.regime == "picture_taking":
            # Forward current frame parameters to the frontend
            self.cutout_coordinates = self.backend.get_current_frame_cutout_coordinates(frame = self.current_frame, frame_size = self.frame_size)
            self.current_label = "Pictures taken: {}".format(len(self.frontend.photos["pictures"]))
            self.frontend.current_label = self.current_label
        else:
            self.show_labels = False
        
        # Forward current frame to the frontend
        self.frontend.current_frame = self.current_frame

    def loop_finalize(self):
        pass

    def loop_interrupt_handler(self):
        pass

    def loop_stop_check(self):
        self.this_is_the_end = False

        if self.video_length is None:
            self.this_is_the_end = False
        elif self.length_is_nframes and self.frontend.current_loop_nr >= self.video_length - 1:
            self.this_is_the_end = True
        elif time() >= self.frontend.start_time + self.video_length:
            self.this_is_the_end = True

        # Checking if regime = picture_taking
        if self.regime == "picture_taking":
            if len(self.frontend.photos["pictures"]) >= self.n_photos and self.this_is_the_end == False:
                self.this_is_the_end = True

        if self.this_is_the_end:
            self.dprint("\tEnd condition met.")
        
        return self.this_is_the_end
    
    def dprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == "__main__":
    labelling_model = KerasDetector(model_specification = "mobilenet", language = "dan-DK")
    videoloop = VideoLoop(model = labelling_model, regime = "object_detection", video_length = None, frontend = "opencv", video_path = "small.mp4")
    videoloop.start()

