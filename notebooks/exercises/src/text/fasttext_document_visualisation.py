from enum import Enum
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import wikipedia
from IPython.display import display, clear_output
from fastText import FastText
from ipywidgets import Tab, Text, Checkbox, Label, HBox, VBox, Button, Dropdown, Textarea
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# TODO: Make system fir handling DisambiguationError when searching wikipedia


_default_wikipedia_articles = [
    "Duck",
    "Fish",
    "Animal",
    "Goose",
    "Elephant"
]


class DocumentVectorTypes(Enum):
    sum = "Sum"
    mean = "Mean"
    sum_over_sqrt = "Sum over root of n"

    @classmethod
    def __iter__(cls):
        yield DocumentVectorTypes.sum
        yield DocumentVectorTypes.mean
        yield DocumentVectorTypes.sum_over_sqrt


_checkbox_layout = dict(width="30pt")
_label_layout = dict(width="180pt")
_remove_layout = dict(width="60pt")
_additional_info_layout = dict(width="380pt", padding="0pt 0pt 0pt 20pt")
_editable_text_layout = dict(width="350pt")
_editable_label_layout = dict(width="50pt")


@lru_cache(maxsize=1000)
def _get_wikipedia_summary(name):
    return wikipedia.summary(name)


@lru_cache(maxsize=1000)
def _get_wikipedia_search(name):
    return wikipedia.search(name)


@lru_cache(maxsize=1000)
def _get_fasttext_document_embedding(fasttext_model, document):
    return fasttext_model.get_sentence_vector(document)


@lru_cache(maxsize=3000)
def _get_fasttext_word_embedding(fasttext_model, word):
    return fasttext_model.get_word_vector(word)


class Document:
    @property
    def name(self):
        raise NotImplementedError

    @property
    def text(self):
        raise NotImplementedError

    @property
    def on(self):
        raise NotImplementedError


class EditableDocument(Document):
    n_rows = 0

    def __init__(self):
        EditableDocument.n_rows += 1
        self._name = "T{}".format(EditableDocument.n_rows)
        self._text = Textarea(
            value='',
            placeholder='Type something',
            description='',
            disabled=False,
            layout=_editable_text_layout,
        )
        self.label = Label(value=self._name, layout=_editable_label_layout)
        self.checkbox = Checkbox(value=True, layout=_checkbox_layout, indent=False)
        self.remove_button = Button(
            description='Remove',
            disabled=False,
            button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
            layout=_remove_layout,
        )

    @property
    def name(self):
        return self._name

    @property
    def text(self):
        return self._text.value

    @property
    def on(self):
        return self.checkbox.value and self.text

    def __iter__(self):
        yield self.checkbox
        yield self.label
        yield self._text
        yield self.remove_button

    @staticmethod
    def make_header():
        return HBox(
            (Label("Use", layout=_checkbox_layout),
             Label("Name", layout=_editable_label_layout),
             Label("Text", layout=_editable_text_layout),
             Label("Remove", layout=_remove_layout),
             )
        )


class WikipediaPageRow(Document):
    def __init__(self, name=None):
        self.editable = name is None
        self.checkbox = Checkbox(value=True, layout=_checkbox_layout, indent=False)
        self.true_name = self.summary = self.last_textfield = None

        if self.editable:
            self.text_field = Text(layout=_label_layout)
        else:
            self.text_field = Label(value=name, layout=_label_layout)

        self.remove_button = Button(
            description='Remove',
            disabled=False,
            button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
            layout=_remove_layout,
        )
        self.additional_info = Label("", layout=_additional_info_layout)
        self.ambiguity_dropdown = None

    def __iter__(self):
        yield self.checkbox
        yield self.text_field
        yield self.remove_button

        if self.ambiguity_dropdown is None:
            yield self.additional_info
        else:
            yield self.ambiguity_dropdown

    def _search(self):
        if self.last_textfield != self.text_field_value:
            try:
                self.last_textfield = self.text_field_value
                self.true_name = _get_wikipedia_search(name=self.text_field_value)[0]
                self.summary = _get_wikipedia_summary(self.true_name)
                self.ambiguity_dropdown = None

            except (wikipedia.DisambiguationError, IndexError) as e:
                if isinstance(e, wikipedia.DisambiguationError):
                    options = [val for val in e.options if not "All pages" in val]

                    self.ambiguity_dropdown = Dropdown(
                        options=options,
                        value=options[0],
                        description='-> Ambiguity:',
                        disabled=False,
                        layout=_additional_info_layout
                    )
                    self.ambiguity_dropdown.observe(self._true_name_from_dropdown)
                    self._true_name_from_dropdown()
                else:
                    self.last_textfield = ""
                    self.true_name = ""
                    self.summary = None
                    self.ambiguity_dropdown = None

    def _true_name_from_dropdown(self, _=None):
        self.true_name = self.ambiguity_dropdown.value
        self.summary = _get_wikipedia_summary(self.true_name)

    @property
    def text_field_value(self):
        return self.text_field.value

    @property
    def name(self):
        self._search()
        return self.true_name

    @property
    def text(self):
        self._search()
        return self.summary

    @property
    def on(self):
        return self.checkbox.value and self.text_field_value

    @staticmethod
    def make_header():
        return HBox(
            (Label("Use", layout=_checkbox_layout),
             Label("Page Search Name", layout=_label_layout),
             Label("Remove", layout=_remove_layout),
             Label("Search Result", layout=_additional_info_layout))
        )


class DocumentRowTab:
    def __init__(self, title, row_type, initial_content=None):
        self.row_type = row_type
        self.title = title
        self.observers = []

        # Make a header
        self._header = self.row_type.make_header()

        # Rows
        self._rows = initial_content if initial_content is not None else []

        # Observe remove buttons
        for page in self._rows:
            page.remove_button.on_click(self._remove_page)

        # Button for adding a new row
        self._new_row_button = Button(
            description='New row',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
        )
        self._new_row_button.on_click(self._add_page)

    def observe(self, func):
        self.observers.append(func)

    def _on_change(self):
        for observer in self.observers:
            observer()

    @property
    def rows(self):
        return self._rows

    @property
    def n_rows(self):
        return len(self._rows)

    @property
    def tab_child(self):
        child = VBox(
            (self._header,) +
            tuple(HBox(tuple(val)) for val in self._rows) +
            (HBox((Label("", layout=dict(width="60pt")), self._new_row_button)),)
        )
        return child

    def _add_page(self, _=None):
        # Make new page
        new_row = self.row_type()
        new_row.remove_button.on_click(self._remove_page)

        # Add
        self._rows.append(new_row)

        # Broadcast change
        self._on_change()

    def _remove_page(self, button):
        # Find the pressed button
        nr = next(idx for idx, row in enumerate(self._rows) if row.remove_button == button)

        # Remove the nr
        self._rows = self._rows[:nr] + self._rows[nr + 1:]

        # Broadcast change
        self._on_change()


class DocumentEmbeddingVisualiser:
    def __init__(self, fasttext_model, figsize=(8, 6)):
        self.figsize = figsize
        self.fig = self.tab = None

        # Get fastText model
        self.fasttext_model = fasttext_model  # type: FastText._FastText

        # Button for plotting document embeddings
        self._do_embeddings_button = Button(
            description="Do Document Embeddings",
            button_style="success",
            layout=dict(width="180pt")
        )
        self._do_embeddings_button.on_click(self._do_embeddings)

        # Choice of document vectors
        self._document_vector_type = Dropdown(
            options=[val.value for val in DocumentVectorTypes],
            value=DocumentVectorTypes.sum.value,
            description='Document Vector Type:',
            disabled=False,
            layout=dict(width="350pt", padding="0pt 0pt 0pt 20pt"),
            style={'description_width': 'initial'},
        )

        # Make Wikipedia-page tab
        self.wikipedia_tab = DocumentRowTab(
            title="Wikipedia Pages",
            row_type=WikipediaPageRow,
            initial_content=[WikipediaPageRow(name=val) for val in _default_wikipedia_articles]
        )
        self.wikipedia_tab.observe(self._display)

        # Free text tab
        self.free_text_tab = DocumentRowTab(
            title="Free Text",
            row_type=EditableDocument,
            initial_content=[EditableDocument()],
        )
        self.free_text_tab.observe(self._display)

        self._display()

    def _display(self):
        clear_output(wait=True)

        # Get current tab index
        tab_index = 0
        if self.tab is not None:
            tab_index = self.tab.selected_index

        # Get Wikipedia tab
        self.tab = Tab()
        self.tab.children = [
            self.wikipedia_tab.tab_child,
            self.free_text_tab.tab_child,
        ]
        self.tab.set_title(0, self.wikipedia_tab.title)
        self.tab.set_title(1, self.free_text_tab.title)

        # Select the wanted tab
        self.tab.selected_index = tab_index

        # Go button with tab
        box = VBox((HBox((self._do_embeddings_button, self._document_vector_type)), self.tab))

        if self.fig is not None:
            # noinspection PyTypeChecker
            display(box, self.fig)
        else:
            # noinspection PyTypeChecker
            display(box)

    def _do_embeddings(self, _=None):
        getting_page_formatter = "Getting Wikipedia pages: {}/{}"

        # Disable button
        self._do_embeddings_button.button_style = "warning"
        self._do_embeddings_button.disabled = True
        self._do_embeddings_button.description = getting_page_formatter.format(0, self.wikipedia_tab.n_rows)

        # Initialize
        document_texts = []
        document_names = []
        document_colors = []

        # Go through wikipedia pages
        for nr, page in enumerate(self.wikipedia_tab.rows):

            # Check if page is wanted and valid
            if page.on:

                # Check search
                if page.name:

                    # Get name and text from wikipedia search
                    document_names.append(page.name)
                    document_texts.append(page.text)
                    document_colors.append("black")

                    # Notify
                    self._do_embeddings_button.description = getting_page_formatter \
                        .format(nr + 1, self.wikipedia_tab.n_rows)
                    page.additional_info.value = "-> " + page.name

                else:
                    page.additional_info.value = "-> UNSUCCESSFUL SEARCH"

        # Go through free text
        for nr, page in enumerate(self.free_text_tab.rows):

            # Check if page is wanted and valid
            if page.on:

                # Get name and text from wikipedia search
                document_names.append(page.name)
                document_texts.append(page.text)
                document_colors.append("blue")

        # Check for not enough samples
        if len(document_texts) >= 3:

            # Compute document vectors
            self._do_embeddings_button.description = "Computing document vectors."
            document_vectors = []
            for summary in document_texts:
                vector_array = np.array([_get_fasttext_word_embedding(self.fasttext_model, word=word) for word in summary])

                # Compute document embedding with mean
                if self._document_vector_type.value == DocumentVectorTypes.mean.value:
                    document_vectors.append(vector_array.mean(0))
                elif self._document_vector_type.value == DocumentVectorTypes.sum.value:
                    document_vectors.append(vector_array.sum(0))
                elif self._document_vector_type.value == DocumentVectorTypes.sum_over_sqrt.value:
                    document_vectors.append(vector_array.sum(0) / np.sqrt(vector_array.shape[0]))
                else:
                    raise ValueError("Unknown document embedding type.")

            # Collect into matrix
            document_vectors = np.array(document_vectors)

            # Fit PCA
            self._do_embeddings_button.description = "Computing 3D projections."
            pca = PCA(n_components=3)
            _ = pca.fit(document_vectors)

            # Get projections
            projection_matrix = pca.components_
            projections = document_vectors.dot(projection_matrix.T)

            # Make plot
            self._do_embeddings_button.description = "Plotting projections."
            self.fig = plt.figure(figsize=self.figsize)
            ax = Axes3D(self.fig)
            for name, vector, color in zip(document_names, projections, document_colors):
                ax.text(
                    *vector,
                    s=name,
                    color=color
                )

            # Set limits
            ax.set_xlim(projections[:, 0].min(), projections[:, 0].max())
            ax.set_ylim(projections[:, 1].min(), projections[:, 1].max())
            ax.set_zlim(projections[:, 2].min(), projections[:, 2].max())

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            plt.close("all")

        # Re-enable button
        self._do_embeddings_button.button_style = "success"
        self._do_embeddings_button.disabled = False
        self._do_embeddings_button.description = "Do Document Embeddings"

        # Refresh
        self._display()
