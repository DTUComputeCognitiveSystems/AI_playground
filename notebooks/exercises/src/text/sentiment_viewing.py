import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from threading import Timer
from src.real_time import TextInputLoop
from src.text.sentiment import SentimentHighlighter
from ipywidgets.widgets import Label, Text, Button, Layout, HBox, VBox, Checkbox

from src.text.utility.text_html import modified_text_to_html


class ViewSentiment:
    def __init__(self):
        plt.close("all")
        self.backend = TextInputLoop(use_widget=True)
        self.highlighter = SentimentHighlighter(self.backend)
        self.backend.add_interface(self.highlighter)
        self.backend.start()

        self.cwd_label = Label(
            value="Working directory: {}".format(Path.cwd()),
            layout=dict(margin="2px 0px 0px 20px"),
        )
        self.save_path = Text(
            value=str(Path("saved_html.html")),
            description='Save path:',
            disabled=False,
            layout=dict(width="50%"),
        )
        self.save_button = Button(
            value=False,
            description='Save to HTML',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='',
            icon='',
            layout=Layout(width='18%', margin="2px 0px 0px 20px"),
        )
        self.progress_label = Label(
            value="",
            layout=dict(margin="2px 0px 0px 20px"),
        )

        self.full_contrast = Checkbox(
            value=False,
            description='Full Contrast',
            disabled=False
        )
        self.do_highlighting = Checkbox(
            value=True,
            description='Do Highlighting',
            disabled=False
        )

        box1 = HBox(
            (self.do_highlighting, self.full_contrast),
            layout=Layout(align_content="flex-start"),
        )

        box2 = HBox(
            (self.save_path, self.save_button),
            layout=Layout(align_content="flex-start"),
        )

        self.storage_dashboard = VBox(
            (box1, self.cwd_label, box2, self.progress_label)
        )

        # Set observers
        self.save_button.on_click(self._save)
        self.full_contrast.observe(self._special_options)
        self.do_highlighting.observe(self._special_options)

        # noinspection PyTypeChecker
        display(self.storage_dashboard)

    def _special_options(self, _=None):
        self.highlighter.full_contrast = self.full_contrast.value
        self.highlighter.do_highlighting = self.do_highlighting.value
        self.highlighter.refresh()

    def _reset_progress(self):
        self.progress_label.value = ""

    def _save(self, _=None):
        if self.highlighter.c_modifiers is not None:
            # Convert to HTML
            html = modified_text_to_html(
                text=self.highlighter.c_text,
                modifiers=self.highlighter.c_modifiers
            )

            # Get save path
            path = Path(self.save_path.value)

            # Save HTML
            with path.open("w") as file:
                file.write(html)

            self.progress_label.value = "Saved!"
            timer = Timer(4.0, self._reset_progress)
            timer.start()
