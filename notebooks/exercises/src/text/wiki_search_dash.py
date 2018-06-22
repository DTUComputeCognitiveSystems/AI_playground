from threading import Timer

from IPython.core.display import display, clear_output
from ipywidgets import Textarea, Checkbox, IntSlider, HBox, VBox, Label
import pandas as pd

from src.text.document_retrieval.wikipedia import Wikipedia


class WikipediaSearchDashboard:
    def __init__(self, wikipedia: Wikipedia):
        self._last_widget_state = None
        self._run_timer = None
        self._last_search = None
        self.wikipedia = wikipedia
        self.text = Textarea(
            value='',
            placeholder='Type something',
            description='Search:',
            disabled=False,
            layout=dict(width="90%"),
        )

        self.show_url = Checkbox(
            value=True,
            description='Show URL',
            disabled=False,
        )
        self.show_abstract = Checkbox(
            value=False,
            description='Show Abstract',
            disabled=False,
        )

        self.n_results = IntSlider(
            value=10,
            step=1,
            description='Number of results:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=dict(width="90%"),
            style=dict(description_width='initial'),
        )

        self.progress_label = Label(
            value=""
        )

        box1 = HBox(
            (self.show_url, self.show_abstract)
        )
        self.dashboard = VBox(
            (self.text, box1, self.n_results, self.progress_label)
        )

        # Set observers
        for observer in [self.show_url, self.show_abstract, self.text, self.n_results]:
            observer.observe(self._widgets_updates)

        # noinspection PyTypeChecker
        display(self.dashboard)

    def _widgets_updates(self, _=None):
        self.progress_label.value = "Updating!"

        # Get widget states
        search_text = self.text.value
        show_url = self.show_url.value
        show_abstract = self.show_abstract.value
        max_results = int(self.n_results.value)
        current_search = (search_text, show_url, show_abstract, max_results)

        # Check if this is a new search
        if self._last_widget_state is None or self._last_widget_state != current_search:
            self._last_widget_state = current_search

            # Reset timer
            if self._run_timer is not None:
                self._run_timer.cancel()
            self._run_timer = Timer(1.0, self._run)
            self._run_timer.start()

    def _run(self, _=None):
        # Get widgets state
        search_text = self.text.value
        show_url = self.show_url.value
        show_abstract = self.show_abstract.value
        n_results = int(self.n_results.value)

        # Check if this is a new search or simply new stuff to show
        if self._last_search != search_text:
            self.progress_label.value = "Searching!"

            # Search through wikipedia
            search_results = self.wikipedia.search(query=search_text)
            self._last_search = search_text

            # Get most probably indices
            self._search_indices = search_results.index.tolist()

        # Get documents
        documents = [self.wikipedia.documents[val] for val in self._search_indices[:n_results]]

        # Output table
        titles = [doc.title for doc in documents]
        if all(["Wikipedia: " in val for val in titles]):
            titles = [val[len("Wikipedia: "):] for val in titles]
        table = [titles]
        header = ["title"]

        # Add content
        if show_url:
            table.append([doc.url for doc in documents])
            header.append("URL")
        if show_abstract:
            table.append([doc.abstract for doc in documents])
            header.append("Abstract")

        # Transpose
        table = list(zip(*table))

        # Make dataframe
        self.table = pd.DataFrame(table, columns=header, index=list(range(1, len(table) + 1)))

        def make_hyperlink(val):
            return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(val, val)

        # Set table style and use 1-indexing
        styles = [
            dict(selector="th", props=[("text-align", "left"), ("font-size", "120%")]),
            dict(selector="td", props=[("text-align", "left")]),
        ]
        table_display = self.table.style.set_table_styles(styles) \
            .format({'URL': make_hyperlink})

        # Clear output and show widgets + results
        clear_output()
        # noinspection PyTypeChecker
        display(self.dashboard)
        # noinspection PyTypeChecker
        display(table_display)
        self.table = self.table

        # Update is done
        self.progress_label.value = ""
