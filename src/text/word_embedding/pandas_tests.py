import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import HBox, VBox, Dropdown

from src.text.word_embedding.fast_text_test import get_test_word_groups, WordGroup, fasttext_projections, \
    visualise_vector_pairs
from src.text.word_embedding.jupyter_selection_table import SelectionTable


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
