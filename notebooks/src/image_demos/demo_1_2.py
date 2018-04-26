from ipywidgets.widgets import Button, Dropdown, Layout, Label, VBox, HBox, IntText, Text

from src.image.image_collection import Image_Collector
from src.image.object_detection.keras_detector import KerasDetector


class VideoTakeDashboard:
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
        
        self.label_names = Text(
            value='',
            placeholder='separated by commas',
            description='Labels',
            disabled=False,
        )
        
        self.num_pictures = IntText(
            value=2.0,
            description='#pictures',
            disabled=False,
            layout=Layout(width='18%')
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
                    ( self.start_button, self.label_names,self.num_pictures, self.select_network),
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

        # Get settings
        num_pictures = self.num_pictures.value
        labels = self.label_names.value.split(',')
        num_objects = len(labels)

        self.collector= Image_Collector(num_pictures=num_pictures, num_objects=num_objects)
        self.collector.run_collector( list_of_labels=labels)

        # Re-enable controls
        self.start_button.disabled = False
        self.select_network.disabled = False

        # Clear output
        self.text.value = ""
