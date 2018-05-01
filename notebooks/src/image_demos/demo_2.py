from ipywidgets.widgets import Button, Dropdown, FloatText, Layout, Label, VBox, HBox, ToggleButton,Text, Checkbox
from matplotlib import pyplot as plt

from src.image.image_collection import ImageCollector, load_data,run_video_recognition
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled import LabelledVideo
import os

class DatasetAquisitionDashboard:
    def __init__(self):

        self.start_button = Button(
            value=False,
            description='Start Camera',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start the camera to take pictures for classifier ',
            icon=''
        )
        self.save_button = Button(
            value=False,
            description='Save images',
            disabled=True,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save images to folder ',
            icon=''
        )
#TODO select image size

        self.save_path = Text(
            value="FILEPATH",
            description='Save path',
            disabled=True
        )
        self.num_pictures = FloatText(
            value=12,
            description='#pictures',
            disabled=False
        )
        self.use_augmentation = Checkbox(
        value=False,
        description='Augmentation',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Use Augmentation',
        icon='check'
    )

        self.text = Label(
            value='',
            layout=Layout(
                justify_content='space-around',
            )
        )

        self.widget_box = VBox(
            (
                HBox(
                    (self.start_button, self.num_pictures,  self.use_augmentation,self.save_path, self.save_button),
                    layout=Layout(justify_content="space-around")
                ),
                self.text
            )
        )

        self.start_button.on_click(self._start_video)
        self.save_button.on_click(self._save_images)

    @property
    def start(self):
        return self.widget_box
    def _save_images(self, _):
        
        self.text.value = "Saving images ..."
        use_augmentation = self.use_augmentation.value
        save_path = os.path.normpath(self.save_path.value)
        self.collector.save_images(save_path, use_augmentation=use_augmentation)
        
        self.text.value = "Done!"
    def _start_video(self, _):
        # Reset start-button and notify
        self.start_button.value = False
        self.text.value = "Starting Video! (please wait)"
        # Disable controls
        self.start_button.disabled = True
        self.num_pictures.disabled = True
        self.use_augmentation.disabled = True
        self.save_path.disabled = True

        # Get settings
        
        num_pictures = int(self.num_pictures.value)

        # Start video
        self.collector= ImageCollector(num_pictures)
        self.collector.run_collector(use_binary = True)
        # Re-enable controls
        self.start_button.disabled = False
        self.num_pictures.disabled = False
        self.save_button.disabled = False
        self.use_augmentation.disabled = False
        self.save_path.disabled = False
        
        self.text.value = ""
        # Clear output
