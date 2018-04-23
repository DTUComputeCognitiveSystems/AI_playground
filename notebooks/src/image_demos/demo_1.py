from ipywidgets.widgets import Button, Dropdown, FloatText, Layout, Label, VBox, HBox
from matplotlib import pyplot as plt

from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled import LabelledVideo


class VideoRecognitionDashboard:
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

        self.video_length = FloatText(
            value=12.0,
            description='Video length:',
            disabled=False
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
                    (self.start_button, self.video_length, self.select_network),
                    layout=Layout(justify_content="space-around")
                ),
                self.text
            )
        )

        self.start_button.on_click(self._start_video)

    @property
    def start(self):
        return self.widget_box

    def _start_video(self, _):
        # Reset start-button and notify
        self.start_button.value = False
        self.text.value = "Starting Video! (please wait)"

        # Disable controls
        self.start_button.disabled = True
        self.select_network.disabled = True
        self.video_length.disabled = True

        # Get settings
        model_name = self.select_network.value
        video_length = self.video_length.value

        # Make network
        net = KerasDetector(model_name=model_name, exlude_animals=True)

        # Start network
        the_video = LabelledVideo(net, video_length=video_length)
        the_video.start()
        plt.show()
        while not the_video.real_time_backend.stop_now:
            plt.pause(.5)

        # Re-enable controls
        self.start_button.disabled = False
        self.select_network.disabled = False
        self.video_length.disabled = False

        # Clear output
        self.text.value = "Done."
