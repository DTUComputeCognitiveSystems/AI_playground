import random
from time import sleep
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import HBox, VBox, Dropdown, FloatSlider, IntSlider, Label, IntText, Button, Checkbox
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from src.text.word_embedding.fast_text_visualisation import get_test_word_groups, WordGroup, fasttext_projections, \
    visualise_vector_pairs
from notebooks.exercises.src.text.jupyter_selection_table import SelectionTable


class WordEmbeddingVisualizer:
    def __init__(self, fasttext_model, figsize=(8, 6)):
        self.figsize = figsize
        self.dataframe = self.dashboard = None
        self.current_group_name = self.current_group = self.current_projection_name = self.fig = None
        self.word_groups = get_test_word_groups()

        # Get fastText model
        self.fasttext_model = fasttext_model

        # Make selector of method for projections.
        self.methods_selector = Dropdown(
            options=["pca", "svd difference"],
            value="pca",
            description='Vector Plane:',
            disabled=False,
        )
        self.methods_selector.observe(self._new_projection)

        # Make selector of word-group
        self.words_selector = Dropdown(
            options=[val.name for val in self.word_groups],
            value=self.word_groups[0].name,
            description='Word Group:',
            disabled=False,
        )
        self.words_selector.observe(self._new_word_group)

        # New (first) word group selected
        self._new_word_group()

    def _new_word_group(self, _=None):
        # Check if any change is made to state of widgets
        if self.current_group_name != self.words_selector.value:
            self.current_group_name = self.words_selector.value

            # Get current word group and compute longest length of row for padding with empty string
            self.current_group = next(val for val in self.word_groups if val.name == self.words_selector.value)
            max_length = max([len(val) for val in self.current_group])

            # Make dataframe with words
            self.dataframe = pd.DataFrame([
                words + [""] * (max_length - len(words)) for words in self.current_group
            ])

            # Get fastText vectors
            self.vectors = np.array([
                [self.fasttext_model.get_word_vector(word) for word in group]
                for group in self.current_group
            ])

            # Make projections
            self.current_projection_name = None
            self._new_projection()

    def _new_projection(self, _=None):
        # Check if actual change was made
        if self.current_projection_name != self.methods_selector.value:
            self.current_projection_name = self.methods_selector.value

            # Compute projections
            self.projected_vectors = fasttext_projections(
                word_group=self.current_group,
                fasttext_model=self.fasttext_model,
                word_group_for_plane=self.current_group,
                method=self.methods_selector.value
            )

            # Make selection table
            self.selection_table = SelectionTable(data=self.dataframe)
            self.selection_table.observe(self.refresh)

            # Refresh view
            self.refresh()

    def refresh(self, _=None):
        # Make dashboard
        self.dashboard = VBox(
            (self.methods_selector, HBox([self.words_selector]), self.selection_table.table),
            layout=dict(width="{}pt".format(self.selection_table.table_width))
        )

        # Plot vectors
        self.plot_vectors()

        # Clear output before next display
        clear_output(wait=True)

        # noinspection PyTypeChecker
        display(self.dashboard, self.fig)

    def plot_vectors(self):
        # Get Word-Group subset
        word_group_subset = WordGroup(
            name=self.current_group.name,
            word_groups=[
                [word for word, word_on in zip(row, self.selection_table.col_on) if word_on]
                for row, row_on in zip(self.current_group, self.selection_table.row_on) if row_on
            ],
            attributes=self.current_group.attributes
        )

        # Get projections subset
        projections_subset = [
                [vector for vector, word_on in zip(row, self.selection_table.col_on) if word_on]
                for row, row_on in zip(self.projected_vectors, self.selection_table.row_on) if row_on
            ]

        # Check if anything was selected (otherwise there is nothing to plot)
        self.fig = None
        if word_group_subset.word_groups:
            lengts = [len(val) for val in word_group_subset.word_groups]
            if sum(lengts) > 1:

                # Visualise the vector pairs
                self.fig = visualise_vector_pairs(
                    vectors=projections_subset,
                    word_group=word_group_subset,
                    figsize=self.figsize
                )


class CompleteWordEmbeddingVisualizer:
    def __init__(self, fasttext_model):
        self.fasttext_model = fasttext_model
        self.n_samples = self.fig = None
        self._hold = False

        # Get vocabulary
        self.vocabulary = self.fasttext_model.get_labels()
        random.shuffle(self.vocabulary)
        self.n_words = len(self.vocabulary)

        # Ticks for slider
        self._n_ticks = 100
        self._ticks = np.logspace(
            start=1, stop=np.log10(self.n_words),
            endpoint=True, base=10, num=self._n_ticks
        )

        # Slider for how many to plot
        self.n_samples_slider = IntSlider(
            value=5,
            min=0,
            max=self._n_ticks-1,
            step=1,
            description='# Samples:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False,
            layout=dict(width="50%"),
        )

        # Label
        self.n_samples_text = IntText(
            value=0,
            layout=dict(width="15%"),
        )

        # Button
        self.button = Button(
            description='Plot points',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=dict(width="15%"),
        )
        self.button.on_click(self._button_clicked)

        # Checkbox for showing words instead of points
        self.show_words = Checkbox(
            value=False,
            description='Show words',
            disabled=False,
            layout=dict(width="15%", padding="0pt 0pt 0pt 10pt"),
            indent=False,
        )
        self.show_words.observe(self._update_button_color)

        # Observations
        self._slider_moved()
        self.n_samples_slider.observe(self._slider_moved)
        self.n_samples_text.observe(self._int_text_edited)

        self.display()

    def display(self):
        clear_output(wait=True)

        box = HBox((self.n_samples_slider, self.n_samples_text, self.button, self.show_words))
        if self.fig is not None:
            # noinspection PyTypeChecker
            display(box, self.fig)
        else:
            # noinspection PyTypeChecker
            display(box)

    def _update_button_color(self, _=None):
        factor = 0.5 if self.show_words.value else 1.0

        if self.n_samples > self._ticks[int(self._n_ticks * factor * 4 / 5)]:
            self.button.button_style = "danger"
        elif self.n_samples > self._ticks[int(self._n_ticks * factor * 2 / 3)]:
            self.button.button_style = "warning"
        elif self.n_samples > self._ticks[int(self._n_ticks * factor * 1 / 2)]:
            self.button.button_style = "info"
        else:
            self.button.button_style = "success"

    def _int_text_edited(self, _=None):
        if not self._hold:
            self._hold = True
            self.n_samples = self.n_samples_text.value

            # Ensure minimum
            if self.n_samples < 10:
                self.n_samples = self.n_samples_text.value = 10

            # Determine tick for slider and ensure maximum
            if self._ticks[-1] > self.n_samples:
                loc = next(idx for idx, val in enumerate(self._ticks) if val > self.n_samples)
            else:
                self.n_samples = self._ticks[-1]
                self.n_samples_text.value = self.n_samples
                loc = self._n_ticks

            self.n_samples_slider.value = loc

        self._update_button_color()
        self._hold = False

    def _slider_moved(self, _=None):
        if not self._hold:
            self._hold = True
            self.n_samples = int(self._ticks[self.n_samples_slider.value])
            self.n_samples_text.value = self.n_samples

        self._update_button_color()
        self._hold = False

    def _disable_button(self):
        self.button.button_style = ""
        self.button.disabled = True

    def _enable_button(self):
        self.button.disabled = False
        self._update_button_color()

    def _button_clicked(self, _=None):
        self._disable_button()
        sleep(0.1)

        # Get a sample of words
        words = self.vocabulary[:self.n_samples]

        # Get vectors
        vectors = np.array([self.fasttext_model.get_word_vector(word) for word in words])

        # Fit PCA
        pca = PCA(n_components=3)
        _ = pca.fit(vectors)

        # Get projections
        projection_matrix = pca.components_
        projections = vectors.dot(projection_matrix.T)

        # Make figure
        plt.close("all")
        self.fig = plt.figure(figsize=(8, 6))
        ax = Axes3D(self.fig)

        if self.show_words.value:
            for loc, word in zip(projections, words):
                ax.text(
                    x=loc[0],
                    y=loc[1],
                    z=loc[2],
                    s=word,
                    ha="center",
                    va="center"
                )
        else:
            ax.scatter(
                xs=projections[:, 0],
                ys=projections[:, 1],
                zs=projections[:, 2],
            )

        # Limits
        ax.set_xlim(projections[:, 0].min(), projections[:, 0].max())
        ax.set_ylim(projections[:, 1].min(), projections[:, 1].max())
        ax.set_zlim(projections[:, 2].min(), projections[:, 2].max())

        # Re-enable button
        self._enable_button()

        self.display()

