import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, HBox, VBox, Label, Dropdown
from ipywidgets import HTML

from src.text.word_embedding.fast_text_test import get_test_word_groups, WordGroup, visualize_word_embeddings


class WordEmbeddingVisualizer:
    def __init__(self, fasttext_model, figsize=(8, 6)):
        self.figsize = figsize
        self.dashboard = None
        self.df = self.row_button_list = self.row_on = self.column_button_list = self.col_on = None
        self.current_group_name = self.current_group = self.fig = None
        self.word_groups = get_test_word_groups()

        # Get fastText model
        self.fasttext_model = fasttext_model

        self._row_height = 28
        self._column_width = 80
        self._row_button_width = 50
        self._table_cell_paddings = 8.16

        self.row_height = "{}pt".format(self._row_height)
        self.column_width = "{}pt".format(self._column_width)
        self.row_button_width = "{}pt".format(self._row_button_width)
        self.table_cell_paddings = "{}pt".format(self._table_cell_paddings)

        self.button_padding = "0pt 0pt 0pt 0pt"

        self.methods_selector = Dropdown(
            options=["pca", "svd"],
            value="pca",
            description='Vector Plane:',
            disabled=False,
        )
        self.words_selector = Dropdown(
            options=[val.name for val in self.word_groups],
            value=self.word_groups[0].name,
            description='Word Group:',
            disabled=False,
        )
        self.words_selector.observe(self._new_word_group)

        self.do_plot = Button(
            description='Plot vectors',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
        )
        self.do_plot.on_click(self.plot_vectors)

        self._new_word_group()

    def _new_word_group(self, _=None):
        if self.current_group_name != self.words_selector.value:
            self.current_group_name = self.words_selector.value

            self.current_group = next(val for val in self.word_groups if val.name == self.words_selector.value)
            max_length = max([len(val) for val in self.current_group])

            self.df = pd.DataFrame([
                words + [""] * (max_length - len(words)) for words in self.current_group
            ])

            # Make a button for each row
            self.row_button_list = [
                Button(
                    description='select',
                    layout=dict(height=self.row_height,
                                width=self.row_button_width,
                                padding=self.button_padding),
                ) for _ in range(self.df.shape[0])
            ]
            self.row_on = [True] * self.df.shape[0]
            for nr, button in enumerate(self.row_button_list):
                button.on_click(self._row_button_clicked(nr=nr))

            # Make a button for each column
            self.column_button_list = [
                Button(
                    description='select',
                    layout=dict(height=self.row_height,
                                width=self.column_width,
                                padding=self.button_padding),
                ) for _ in range(self.df.shape[1])
            ]
            self.col_on = [True] * self.df.shape[1]
            for nr, button in enumerate(self.column_button_list):
                button.on_click(self._col_button_clicked(nr=nr))

            self.dashboard_width = max((self.df.shape[1] + 1) * self._column_width + self._row_button_width, 500)
            self.refresh()

    def _make_table(self):
        # Global styles
        styles = [
            dict(selector="tr", props=[
                ("height", self.row_height),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
            dict(selector="td", props=[
                ("padding-bottom", self.table_cell_paddings),
                ("padding-top", self.table_cell_paddings),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
            dict(selector="th", props=[
                ("padding-bottom", self.table_cell_paddings),
                ("padding-top", self.table_cell_paddings),
                ("vertical-align", "middle"),
                ("text-align", "center"),
                ("width", self.column_width),
            ]),
        ]

        # Make styler for DataFrame
        styler = self.df.style.set_table_attributes('class="table"') \
            .set_table_styles(styles) \
            .apply(self.selection_colorize, axis=None)

        # Render HTML table from DataFrame
        html = styler.render()

        # Build visual components
        v_row_buttons = VBox(
            [Label(description='', layout=dict(height=self.row_height,
                                               width=self.row_button_width,
                                               padding=self.button_padding)),
             *self.row_button_list],
        )
        v_bottom = HBox([
            v_row_buttons,
            HTML(html)
        ])
        v_col_buttons = HBox(
            [Label(description='', layout=dict(height=self.row_height,
                                               width=self.row_button_width,
                                               padding=self.button_padding)),
             Label(description='', layout=dict(height=self.row_height,
                                               width=self.column_width,
                                               padding=self.button_padding)),
             *self.column_button_list],
        )

        # Make dashboard
        self.dashboard = VBox(
            (self.methods_selector, HBox([self.words_selector, self.do_plot]), v_col_buttons, v_bottom,),
            layout=dict(width="{}pt".format(self.dashboard_width))
        )

    def refresh(self):
        self._make_table()
        clear_output(wait=True)

        # noinspection PyTypeChecker
        if self.fig is not None:
            display(self.dashboard, self.fig)
        else:
            display(self.dashboard)

    def _row_button_clicked(self, nr):
        def set_bool(_=None):
            self.row_on[nr] = not self.row_on[nr]
            self.refresh()

        return set_bool

    def _col_button_clicked(self, nr):
        def set_bool(_=None):
            self.col_on[nr] = not self.col_on[nr]
            self.refresh()

        return set_bool

    def selection_colorize(self, _=None):
        formatters = pd.DataFrame([
            ["color: black" if row and col else "color: lightgrey" for col in self.col_on]
            for row in self.row_on
        ])
        return formatters

    def plot_vectors(self, _=None):
        # Get Word-Group subset
        new_word_group = WordGroup(
            name=self.current_group.name,
            word_groups=[
                [word for word, word_on in zip(row, self.col_on) if word_on]
                for row, row_on in zip(self.current_group, self.row_on) if row_on
            ],
            attributes=self.current_group.attributes
        )

        plt.close("all")
        self.fig = None
        if new_word_group.word_groups:
            lengts = [len(val) for val in new_word_group.word_groups]
            if sum(lengts) > 1:

                method = self.methods_selector.value

                self.fig = visualize_word_embeddings(
                    word_group=new_word_group,
                    fasttext_model=self.fasttext_model,
                    word_group_for_plane=self.current_group,
                    method=method,
                    figsize=self.figsize,
                )
                plt.close("all")

        self.refresh()
