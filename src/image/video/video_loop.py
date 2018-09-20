""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from src.image.object_detection.models_base import ImageLabeller
from src.image.object_detection.keras_detector import KerasDetector

from src.real_time.frontend_opencv import OpenCVFrontendController
from src.image.backend_opencv import OpenCVBackendController

from time import time

class VideoLoop:
    def __init__(self, model, store_predictions=False,
                 frame_rate=24, stream_type="thread",
                 video_length=3, length_is_nframes=False,
                 record_frames=False,
                 title="Video", 
                 verbose=False,
                 crosshair_size=(224, 224),
                 backend="opencv", frontend="opencv", video_path = None):
        # Video settings
        self.frame_rate = frame_rate
        self.stream_type = stream_type
        self.record_frames = record_frames
        self.frame_rate = frame_rate
        self.frame_time = int(1000 * 1. / frame_rate)
        self.title = title
        self.verbose = verbose
        self.crosshair_size = crosshair_size
        # Model settings
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

        
        # Getting the backend object
        if backend == "opencv":
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate)
        else:
            self.backend = OpenCVBackendController(frame_rate = self.frame_rate)

        # Getting the frontend object
        if frontend == "opencv":
            self.frontend = OpenCVFrontendController()
        else:
            self.frontend = OpenCVFrontendController()
        
    def run(self):

        self.loop_initialize()
        self.loop_step()
        self.loop_finalize()

        # Get photo
        self.current_frame = self.backend.current_frame
        self.current_frame_time = time()
        self.frame_size = self.current_frame.shape

    def loop_initialize(self):
        self.current_frame = self.backend.current_frame
        self.current_frame_time = time()
        self.frame_size = self.current_frame.shape
        
        # Determine coordinates of cutout for grapping part of frame
        self.current_frame_cutout_coordinates = self.calculateFrameCutout(frame_size=self.frame_size, size=self.crosshair_size).coordinates
        
    
    def loop_step(self):

        # Get frame cutout coordinates
        start_x, start_y, width, height = self.current_frame_cutout_coordinates
        end_x, end_y = start_x + width, start_y + height

        # Get first frame cutout
        self.current_frame_cutout = self.current_frame[start_x:end_x, start_y:end_y]
        
        # Get labels and probabilities for the frame cutout
        labels, probabilities = self.model.label_frame(frame=self.current_frame_cutout)

        # Set analysis
        self.current_label = labels[0]
        self.current_label_probability = probabilities[0]

        # Show crosshair

        # Show label

        # Display

        # Store cut frames if stated
        if self.record_frames:
            self.cut_frames.append(self.current_frame_cutout)
        
         # Storage
        if self.store_predictions:
            self.predictions.append((
                self.current_frame_time, labels, probabilities
            ))

        # Get next image
        self.current_frame = self.backend.current_frame
        self.current_frame_time = time()

    def loop_finalize(self):
        pass
    
    def calculateFrameCutout(self, frame_size, size=None, width_ratio=0.5, height_ratio=None):
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
