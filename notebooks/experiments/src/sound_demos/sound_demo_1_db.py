from notebooks.experiments.src.sound_demos.multilabel_classifier import Recorder
from src.audio.mini_recorder import miniRecorder
import librosa
import pyaudio
from ipywidgets.widgets import Layout, Label, HBox, VBox, HTML, Dropdown, Button, Output
import numpy as np
import os
import sys
import glob
import time

WAV_DIR = "tmp"


class SoundDemo1Dashboard1:
    def __init__(self):
        """
        :param int i: 
        """

        self.select_training_data_options = None

        self.select_training_data = Dropdown(
            options={'Record training data': 0,
                     'Load training data files': 1
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
                     '4': 4
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
        self.play_button.on_click(self._playback)

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

    def _playback(self, v):

        if self.data:
            # instantiate pyaudio
            p = pyaudio.PyAudio()
            i = 0
            for d in enumerate(self.data[0]):
                i = i + 1
                # instantiate stream
                stream = p.open(format=pyaudio.paFloat32, # float32 insteda of paInt16 because librosa loads in float32
                                     channels=1,
                                     rate=22050,
                                     output=True,
                                     )
                print("Playing example {}, label {} ...".format(i, self.data[1][i - 1]))
                time.sleep(1)
                try:
                    stream.write(self.data[0][i - 1], len(self.data[0][i - 1]))
                except Exception:
                    print("Error:", sys.exc_info()[0])
            stream.stop_stream()
            stream.close()
            p.terminate()
        else:
            print("No training data has been loaded. Please load the training data first.")

    def _run(self, v):
        self.prefix = "training"
        self.recorder = Recorder(n_classes=self.select_nclasses.value,
                                 n_files=self.select_nfiles.value,
                                 prefix=self.prefix,
                                 wav_dir='tmp')
        if self.select_training_data.value != 0:
            # Remove the data with the same prefix
            print("Loading the recorded data..")
            try:
                files = self.recorder.get_files()
                s = [True if self.prefix in file else False for file in files]
            except Exception:
                print("Loading failed. No files were found.")
                return
            if True in s:
                self.data = self.recorder.create_dataset()
            else:
                print("Loading failed. No relevant files were found.")
                return

        # If we choose to record the data, then we record it here
        if self.select_training_data.value == 0:
            print("Recording the training data..")
            # Remove the data with the same prefix
            files = self.recorder.get_files()
            for file in files:
                if self.prefix in file:
                    os.remove(file)
            # Start recording
            self.recorder.record(seconds=self.select_nseconds.value, clear_output=False)
            self.data = self.recorder.create_dataset()

        print("Data has been loaded.")

        return


class SoundDemo1Dashboard2:
    def __init__(self):
        """
        :param int i: 
        """
        self.test_sound = None
        self.test_data = None
        self.select_test_data_options = None

        self.select_test_data = Dropdown(
            options={'Record test data': 0,
                     'Load test data files': 1
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
        self.play_button.on_click(self._playback)

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

    def _playback(self, v):
        if self.test_sound is not None:
            # Instantiate miniRecorder
            self.rec = miniRecorder(seconds=1.5)
            self.rec.data = self.test_data
            self.rec.sound = self.test_sound
            if self.test_data is None:
                self.rec.playback(format=pyaudio.paFloat32)
            else:
                self.rec.playback()
        else:
            print("No test sound has been loaded. Please load the test sound first.")

    def _run(self, v):
        self.prefix = "test"
        self.test_filename = "test.wav"
        self.test_sound = None
        self.test_data = None

        # If we choose to record the data, then we record it here
        if self.select_test_data.value == 0:

            # Remove the data with the same prefix
            file_ext = self.prefix + "*.wav"
            files = glob.glob(os.path.join(WAV_DIR, file_ext))
            for file in files:
                if self.prefix in file:
                    os.remove(file)
            # Instantiate miniRecorder
            self.rec = miniRecorder(seconds=1.5)
            # Start recording
            _ = self.rec.record()
            self.test_sound = self.rec.sound
            self.test_data = self.rec.data
            self.rec.write2file(fname=os.path.join(WAV_DIR, self.test_filename))
        else:
            file = glob.glob(os.path.join(WAV_DIR, self.test_filename))
            if file:
                try:
                    (sound_clip, fs) = librosa.load(os.path.join(WAV_DIR, self.test_filename), sr=22050)
                    self.test_sound = np.array(sound_clip)
                    print("File loaded successfully.")
                except Exception:
                    print("File loading has not succeeded.")
            else:
                print("No file named {} has been found in {}".format(self.test_filename, WAV_DIR))

        return

