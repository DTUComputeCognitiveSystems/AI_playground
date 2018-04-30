from ipywidgets.widgets import Button, Dropdown, FloatText, Layout, Label, VBox, HBox, Checkbox, RadioButtons, Text
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'..')
import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent.resolve())
    
#%%    

from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled import LabelledVideo
#%%

class VideoRecognitionDashboard:
    # TODO: Perhaps this could be automated to take any of the video classes and use interact() from IPython to
    # TODO:     generate widgets?

    def __init__(self):

        self.start_button = Button(
            value=False,
            description='Start Camera',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start the camera and the recognition algorithm.',
            icon=''
        )

        self.select_network = Dropdown(
            options=KerasDetector.available_models,
            value=KerasDetector.available_models[0],
            description='Algorithm:',
            disabled=False,
        )
        
        self.use_recorded= RadioButtons(
    options=['MP4', 'Webcam'],
    
    value='MP4',
    description='Input',
    disabled=False,
            layout=Layout(width='20%')
)
        self.video_path = Text(
            value='',
            placeholder='Video path',
            description='Video path',
            disabled=False,
        )

        self.video_length = FloatText(
            value=12.0,
            description='Video length:',
            disabled=True,
            layout=Layout(width='15%')
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
                    (self.start_button, self.use_recorded, self.video_path, self.video_length, self.select_network),
                    layout=Layout(justify_content="space-around")
                ),
                self.text
            )
        )

        self.start_button.on_click(self._start_video)
        self.use_recorded.on_trait_change(self.refresh_state)

    @property
    def start(self):
        return self.widget_box
    def refresh_state(self):
        use_recorded = self.use_recorded.value == "MP4"

        self.video_length.disabled = use_recorded
        self.video_path.disabled = not use_recorded
        self.video_length.disabled = use_recorded

    def _start_video(self, _):
        # Reset start-button and notify
        self.start_button.value = False
        self.text.value = "Starting Video! (please wait)"

        # Disable controls
        self.start_button.disabled = True
        self.select_network.disabled = True
        self.video_length.disabled = True
        self.use_recorded.disabled = True
        self.video_path.disabled = True

        # Get settings
        model_name = self.select_network.value
        video_length = self.video_length.value

        # Make network
        net = KerasDetector(model_name=model_name, exlude_animals=True)

        # Start network
        video_path = None
        if self.use_recorded.value == "MP4":
            video_path = self.video_path.value
            video_length =120
        the_video = LabelledVideo(net, video_length=video_length,video_path = video_path)
        the_video.start()
        plt.show()
        while not the_video.real_time_backend.stop_now:
            plt.pause(.5)

        # Re-enable controls
        self.start_button.disabled = False
        self.select_network.disabled = False
        self.video_length.disabled = False
        self.use_recorded.disabled = False
        self.video_path.disabled = False


        # Clear output
        self.text.value = "Done."
