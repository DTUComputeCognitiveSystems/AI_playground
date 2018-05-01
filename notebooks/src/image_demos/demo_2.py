import os
from pathlib import Path

from ipywidgets.widgets import Button, FloatText, Layout, Label, VBox, HBox, Text, Checkbox

from src.image.image_collection import ImageCollector


class TwoClassCameraDashboard:
    def __init__(self):
        self.start_button = Button(
            value=False,
            description='Start Camera',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start the camera to take pictures for classifier ',
            icon='',
            layout=Layout(width='25%'),
        )

        self.save_button = Button(
            value=False,
            description='Save images',
            disabled=True,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save images to folder ',
            icon='',
            layout=Layout(width='18%', margin="2px 0px 0px 20px"),
        )

        self.save_path = Text(
            value=str(Path("path", "to", "directory")),
            description='Save path:',
            disabled=False,
        )

        self.num_pictures = FloatText(
            value=12,
            description='#pictures:',
            disabled=False,
            layout=Layout(width='15%'),
        )

        self.use_augmentation = Checkbox(
            value=False,
            description='Augmentation',
            disabled=False,
            tooltip='Use Augmentation',
            icon='check',
        )

        self.progress_text = Label(
            value='',
            layout=Layout(margin="5px 0px 5px 20px"),
        )

        self.widget_box = VBox((
            HBox(
                (self.num_pictures, self.use_augmentation, self.start_button),
                layout=Layout(align_content="flex-start"),
            ),
            HBox(
                (self.progress_text,),
                layout=Layout(align_content="flex-start"),
            ),
            HBox(
                (self.save_path,self.save_button),
                layout=Layout(align_content="flex-start"),
            ),
        ))

        self.start_button.on_click(self._start_video)
        self.save_button.on_click(self._save_images)

    @property
    def start(self):
        return self.widget_box

    def _save_images(self, _):
        # Get values from widgets
        use_augmentation = self.use_augmentation.value
        save_path = Path(self.save_path.value)

        self.progress_text.value = "Saving images to: {}".format(save_path)

        # Use collector to save images
        self.collector.save_images(save_path, use_augmentation=use_augmentation)

        self.progress_text.value = "Images saved to {}".format(save_path)

    def _start_video(self, _):
        # Reset start-button and notify
        self.start_button.value = False
        self.progress_text.value = "Starting camera! (please wait)"

        # Disable controls
        self.start_button.disabled = True
        self.num_pictures.disabled = True
        self.use_augmentation.disabled = True
        self.save_path.disabled = True

        # Get settings
        num_pictures = int(self.num_pictures.value)

        # Start video
        self.collector = ImageCollector(num_pictures)
        self.collector.run_collector(use_binary=True)

        # Re-enable controls
        self.start_button.disabled = False
        self.num_pictures.disabled = False
        self.save_button.disabled = False
        self.use_augmentation.disabled = False
        self.save_path.disabled = False

        self.progress_text.value = ""
