from functools import lru_cache
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from IPython.display import display, clear_output
from fastText import FastText
from ipywidgets import Tab, Text, Checkbox, Label, HBox, VBox, Button, Dropdown
import wikipedia
from sklearn.decomposition import PCA

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
_additional_info_layout = dict(width="180pt", padding="0pt 0pt 0pt 20pt")


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


class WikipediaPageRow(Document):
    def __init__(self, name=None):
        self.editable = name is None
        self.checkbox = Checkbox(value=True, layout=_checkbox_layout, indent=False)
        self.true_name = self.summary = None

        if self.editable:
            self.text_field = Text(layout=_label_layout)
        else:
            self.text_field = Label(value=name, layout=_label_layout)

        self.button = Button(
            description='Remove',
            disabled=False,
            button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
            layout=_remove_layout,
        )
        self.additional_info = Label("", layout=_additional_info_layout)

    def __iter__(self):
        yield self.checkbox
        yield self.text_field
        yield self.button
        yield self.additional_info

    def _search(self):
        if self.true_name is None:
            self.true_name = _get_wikipedia_search(name=self.name)[0]
            self.summary = _get_wikipedia_summary(self.true_name)

    @property
    def searched_name(self):
        self._search()
        return self.true_name

    @property
    def name(self):
        return self.text_field.value

    @property
    def text(self):
        self._search()
        return self.summary

    @property
    def on(self):
        return self.checkbox.value and self.name


class WikipediaPagesTab:
    def __init__(self):
        self.observers = []

        # Make a header
        self._header = HBox(
            (Label("Use", layout=_checkbox_layout),
             Label("Page Search Name", layout=_label_layout),
             Label("Remove", layout=_remove_layout),
             Label("Search Result", layout=_additional_info_layout))
        )

        # Wikipedia pages
        self._wikipedia_pages = [WikipediaPageRow(name=val) for val in _default_wikipedia_articles]

        # Observe remove buttons
        for page in self._wikipedia_pages:
            page.button.on_click(self._remove_page)

        # Button for adding a new page
        self._new_page_button = Button(
            description='New page',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
        )
        self._new_page_button.on_click(self._add_page)

    def observe(self, func):
        self.observers.append(func)

    def _on_change(self):
        for observer in self.observers:
            observer()

    @property
    def pages(self):
        return self._wikipedia_pages

    @property
    def n_pages(self):
        return len(self._wikipedia_pages)

    @property
    def tab(self):
        tab = Tab()
        tab.children = [
            VBox(
                (self._header,) +
                tuple(HBox(tuple(val)) for val in self._wikipedia_pages) +
                (HBox((Label("", layout=dict(width="60pt")), self._new_page_button)),)
            )
        ]
        tab.set_title(0, "Wikipedia Pages")
        return tab

    def _add_page(self, _=None):
        # Make new page
        new_row = WikipediaPageRow()
        new_row.button.on_click(self._remove_page)

        # Add
        self._wikipedia_pages.append(new_row)

        # Broadcast change
        self._on_change()

    def _remove_page(self, button):
        # Find the pressed button
        nr = next(idx for idx, page in enumerate(self._wikipedia_pages) if page.button == button)

        # Remove the nr
        self._wikipedia_pages = self._wikipedia_pages[:nr] + self._wikipedia_pages[nr + 1:]

        # Broadcast change
        self._on_change()


class DocumentEmbeddingVisualiser:
    def __init__(self, fasttext_model, figsize=(8, 6)):
        self.figsize = figsize
        self.fig = None

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
        self.wikipedia_tab = WikipediaPagesTab()
        self.wikipedia_tab.observe(self._display)

        self._display()

    def _display(self):
        clear_output(wait=True)

        # Get Wikipedia tab
        tab = self.wikipedia_tab.tab

        # Go button with tab
        box = VBox((HBox((self._do_embeddings_button, self._document_vector_type)), tab))

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
        self._do_embeddings_button.description = getting_page_formatter.format(0, self.wikipedia_tab.n_pages)

        # Download pages
        summaries = []
        summary_names = []
        for nr, page in enumerate(self.wikipedia_tab.pages):

            # Check if page is wanted and valid
            if page.on:

                # Get name and text from wikipedia search
                summary_names.append(page.searched_name)
                summaries.append(page.text)

                # Notify
                self._do_embeddings_button.description = getting_page_formatter \
                    .format(nr + 1, self.wikipedia_tab.n_pages)
                page.additional_info.value = "-> " + page.searched_name

        # Check for not enough samples
        if len(summaries) >= 3:

            # Go through pages and computer mean word embedding
            self._do_embeddings_button.description = "Computing document vectors."
            document_vectors = []
            for summary in summaries:
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
            for name, vector in zip(summary_names, projections):
                ax.text(
                    *vector,
                    s=name
                )

            # Set limits
            ax.set_xlim(projections[:, 0].min(), projections[:, 0].max())
            ax.set_ylim(projections[:, 1].min(), projections[:, 1].max())
            ax.set_zlim(projections[:, 2].min(), projections[:, 2].max())

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        # Re-enable button
        self._do_embeddings_button.button_style = "success"
        self._do_embeddings_button.disabled = False
        self._do_embeddings_button.description = "Do Document Embeddings"

        # Refresh
        self._display()
