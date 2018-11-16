from notebooks.experiments.src.sound_demos.multilabel_classifier import Recorder
from src.audio.mini_recorder import miniRecorder
from ipywidgets.widgets import Layout, Label, HBox, VBox, HTML, Dropdown, Button, Output
import os
import glob

WAV_DIR = "tmp"


class SoundDemo1Dashboard1:
    def __init__(self):
        """
        :param int i: 
        """

        self.select_training_data_options = None

        self.select_training_data = Dropdown(
            options={'Record training data': 0,
                     'Load file ...': 1
                     },
            value=0,
            description='Choose training data:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.select_training_data.observe(self.on_change)

        self.select_nseconds = Dropdown(
            options={
                '2s': 2,
                '3s': 3,
                '5s': 5
            },
            value=2,
            description='Choose recording length:',
            disabled=False,
            layout=Layout(width='400px', display='block'),
            style={'description_width': '160px'},
        )

        self.select_nclasses = Dropdown(
            options={'2': 2,
                     '3': 3,
                     '4': 4
                     },
            value=2,
            description='Choose number of classes:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.select_nfiles = Dropdown(
            options={'12': 12,
                     '8': 8,
                     '6': 6,
                     '4': 4,
                     '2': 2
                     },
            value=12,
            description='Choose number of files:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.container = Output(
            value="",
        )

        self.submit_button = Button(
            value=False,
            description='Get data',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='TRAIN: use recorded sounds or record new',
            icon=''
        )

        self.play_button = Button(
            value=False,
            description='Play the recording',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='TRAIN: play recorded or loaded sounds',
            icon=''
        )

        self.widget_box = VBox(
            (
                HBox(
                    (
                        VBox((
                            self.select_training_data,
                            self.select_nseconds,
                            self.select_nclasses,
                            self.select_nfiles
                        ), ),
                        VBox((
                            self.submit_button,
                            self.play_button
                        ), )
                    ),
                ),
                self.container
            ),
        )

        self.submit_button.on_click(self._run)

    @property
    def start(self):
        return self.widget_box

    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if change["new"] != 0:
                self.select_nseconds.layout.display = 'none'
                self.select_nseconds.disabled = True
            else:
                self.select_nseconds.layout.display = 'block'
                self.select_nseconds.disabled = False
            # print("changed to {}".format(change['new']))

    def _run(self, v):
        self.prefix = "training"
        self.recorder = Recorder(n_classes=self.select_nclasses.value,
                                 n_files=self.select_nfiles.value,
                                 prefix=self.prefix,
                                 wav_dir='tmp')

        # If we choose to record the data, then we record it here
        if self.select_training_data.value == 0:
            # Remove the data with the same prefix
            files = self.recorder.get_files()
            for file in files:
                if self.prefix in file:
                    os.remove(file)
            # Start recording
            self.recorder.record(seconds=self.select_nseconds.value, clear_output=False)

        self.data = self.recorder.create_dataset()
        return


class SoundDemo1Dashboard2:
    def __init__(self):
        """
        :param int i: 
        """
        self.select_test_data_options = None

        self.select_test_data = Dropdown(
            options={'Record training data': 0,
                     'Load file ...': 1
                     },
            value=0,
            description='Choose training data:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.select_test_data.observe(self.on_change)

        self.select_nseconds = Dropdown(
            options={
                '2s': 2,
                '3s': 3,
                '5s': 5
            },
            value=2,
            description='Choose recording length:',
            disabled=False,
            layout=Layout(width='400px', display='block'),
            style={'description_width': '160px'},
        )

        self.container = Output(
            value="",
        )

        self.submit_button = Button(
            value=False,
            description='Get test data',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='TEST: use recorded sounds or record new',
            icon=''
        )

        self.play_button = Button(
            value=False,
            description='Play the recording',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='TEST: play recorded or loaded sounds',
            icon=''
        )

        self.widget_box = VBox(
            (
                HBox(
                    (
                        VBox((
                            self.select_test_data,
                            self.select_nseconds
                        ), ),
                        VBox((
                            self.submit_button,
                            self.play_button
                        ), )
                    ),
                ),
                self.container
            ),
        )

        self.submit_button.on_click(self._run)

    @property
    def start(self):
        return self.widget_box

    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if change["new"] != 0:
                self.select_nseconds.layout.display = 'none'
                self.select_nseconds.disabled = True
            else:
                self.select_nseconds.layout.display = 'block'
                self.select_nseconds.disabled = False
            # print("changed to {}".format(change['new']))

    def _run(self, v):
        self.prefix = "test"

        # If we choose to record the data, then we record it here
        if self.select_test_data.value == 0:
            # Remove the data with the same prefix
            file_ext = self.prefix + "*.wav"
            files = glob.glob(os.path.join(WAV_DIR, file_ext))
            for file in files:
                if self.prefix in file:
                    os.remove(file)
            # Start recording
            rec = miniRecorder(seconds=1.5)
            _ = rec.record()
            rec.write2file("test.wav")

        # self.data = self.recorder.create_dataset()
        return
