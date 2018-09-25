from pathlib import Path

from ipywidgets.widgets import Button, Dropdown, FloatText, Layout, Label, VBox, HBox, RadioButtons, Text
from matplotlib import pyplot as plt

from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled_video import LabelledVideo
from src.image.video.video_loop import VideoLoop

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

        self.use_recorded = RadioButtons(
            options=['Webcam', 'MP4'],
            value='Webcam',
            description='Input',
            disabled=False,
            layout=Layout(width='320px'),
            style={'description_width': '150px'},
        )

        self.video_path = Text(
            value='',
            placeholder=str(Path("path", "to", "video.mp4")),
            description='Video path:',
            disabled=True,
            layout=Layout(width='320px'),
            style={'description_width': '150px'},
        )

        self.select_network = Dropdown(
            options=KerasDetector.available_models,
            value=KerasDetector.available_models[0],
            description='Algorithm:',
            disabled=False,
            layout=Layout(width='220px'),
            style={'description_width': '100px'},
        )

        self.select_frontend = Dropdown(
            options=["opencv","matplotlib"],
            value="opencv",
            description='Frontend:',
            disabled=False,
            layout=Layout(width='220px', padding='8px 0 0 0'),
            style={'description_width': '100px'},
        )

        self.video_length = FloatText(
            value=12.0,
            description='Video length [s]:',
            disabled=False,
            layout=Layout(width='250px'),
            style={'description_width': '130px'},
        )

        self.select_framerate = Dropdown(
            options=[3, 5, 10, 15, 24],
            value=15,
            description='Frame rate:',
            disabled=False,
            layout=Layout(width='250px', padding='8px 0 0 0'),
            style={'description_width': '130px'},
        )

        self.static_text = Label(
            value="Working directory: {}".format(Path.cwd()),
            layout=Layout(margin='30px 0 0 0'),
        )
        self.progress_text = Label(
            value='',
        )
        self.text_box = VBox(
            (self.static_text, self.progress_text),
            layout=Layout(justify_content="flex-start")
        )

        self.widget_box = VBox(
            (
            HBox(
                    (
                        VBox(
                            (self.start_button,),
                            layout=Layout(justify_content="space-around")
                        ),
                        VBox(
                            (self.use_recorded, self.video_path),
                            layout=Layout(justify_content="space-around")
                        ),
                        VBox(
                            (self.video_length, self.select_framerate),
                            layout=Layout(justify_content="space-around")
                        ),
                        VBox(
                            (self.select_network, self.select_frontend),
                            layout=Layout(justify_content="space-around")
                        )
                        
                    ), 
                ),
            HBox(
                (self.text_box,),
                layout=Layout(justify_content="flex-start")
                )
            ),
        )

        self.start_button.on_click(self._start_video)
        self.use_recorded.observe(self.refresh_state)

    @property
    def start(self):
        return self.widget_box

    def refresh_state(self, _=None):
        use_recorded = self.use_recorded.value == "MP4"

        self.video_length.disabled = use_recorded
        self.video_path.disabled = not use_recorded
        self.video_length.disabled = use_recorded

    def _start_video(self, _):
        # Reset start-button and notify
        self.start_button.value = False
        self.progress_text.value = "Starting Video! (please wait)"

        # Disable controls
        self.start_button.disabled = True
        self.select_network.disabled = True
        self.video_length.disabled = True
        self.use_recorded.disabled = True
        self.video_path.disabled = True

        # Get settings
        model_name = self.select_network.value
        video_length = self.video_length.value
        selected_frontend = self.select_frontend.value
        selected_framerate = self.select_framerate.value

        # Make network
        net = KerasDetector(model_specification=model_name, exlude_animals=True)

        # Start network
        video_path = None
        if self.use_recorded.value == "MP4":
            video_path = self.video_path.value
            video_length = 120
        #the_video = LabelledVideo(net, backend=selected_backend, frame_rate = selected_framerate, video_length=video_length, video_path=video_path)
        the_video = VideoLoop(net, frontend = selected_frontend, frame_rate = selected_framerate, video_length=video_length, video_path=video_path)
        self.progress_text.value = "Video running!"
        the_video.start()

        # Re-enable controls
        self.start_button.disabled = False
        self.select_network.disabled = False
        self.video_length.disabled = False
        self.use_recorded.disabled = False
        self.video_path.disabled = False

        # Clear output
        self.progress_text.value = "Done."
