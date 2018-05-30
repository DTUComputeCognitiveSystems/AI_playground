import pickle
from pathlib import Path
from time import time
from xml.etree.cElementTree import Element, fromstring

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.utility.files import ensure_directory, raw_buffered_line_counter

_data_dir = Path("data")
ensure_directory(_data_dir)


# TODO: Perhaps change loading system


class WikipediaDocument:
    def __init__(self, title=None, url=None, abstract=None, text=None):
        self.title = title
        self.url = url
        self.abstract = abstract
        self.text = text

    def __str__(self):
        return "Document('{}')".format(self.title)

    def __repr__(self):
        return str(self)


class WikipediaSearcher:
    __default_wikipedia_path = Path(_data_dir, "wikipedia_en.xml")
    __parsed_documents_path = Path(_data_dir, "parsed_documents.p")
    __processed_documents_path = Path(_data_dir, "processed_documents.p")
    __fields_of_interest = ["title", "abstract", "url"]
    __out_start_tag = "<doc>"
    __out_end_tag = "</doc>"

    def __init__(self, max_lines=None, always_load=False):
        self.documents = None  # type: list[WikipediaDocument]
        self._start_time = time()

        # Check whether processed documents can be used
        use_processed = self._use_processed(always_load=always_load)

        # If we have the processed storage then load
        # TODO: Rename "processed" to "vectorized"
        if use_processed:
            print("\tLoading processed documents.")
            self._processed_storage = pickle.load(WikipediaSearcher.__processed_documents_path.open("rb"))

        # Check whether parsed documents can be used
        use_preparsed = self._use_prepared(always_load=always_load)

        # If preparsed documents are to be used, then get these
        if use_preparsed:
            print("\tLoading preparsed documents.")
            self.documents = pickle.load(WikipediaSearcher.__parsed_documents_path.open("rb"))

        # Otherwise start processing a Wikipedia file
        else:

            # Get wikipedia data path
            self._wiki_data_dir = self._request_wikipedia_path()
            if self._wiki_data_dir is None:
                return

            # Parse Wikipedia file
            self._parse_wikipedia(max_lines=max_lines)

            # Store data
            pickle.dump(self.documents, WikipediaSearcher.__parsed_documents_path.open("wb"))

            # Process documents
            self._process_parsed_documents()

            # Store processed data
            pickle.dump(self._processed_storage, WikipediaSearcher.__processed_documents_path.open("wb"))

        # Progress
        print("WikipediaSearcher ready.")

    @property
    def _processed_storage(self):
        return (self.n_documents,
                self.n_words,
                self.matrix_doc_term,
                self.document_lengths,
                self._avg_document_length,
                self.idf,
                self.term_vectorizer)

    @_processed_storage.setter
    def _processed_storage(self, values):
        (self.n_documents, self.n_words, self.matrix_doc_term, self.document_lengths, self._avg_document_length,
         self.idf, self.term_vectorizer) = values

    def _process_parsed_documents(self):
        # Start
        print("\nStarting processing of parsed documents.")

        print("\t{:8.2f}s: Getting abstracts of documents.".format(time() - self._start_time))

        # Get titles and abstracts
        abstracts = [val.abstract if val.abstract is not None else "" for val in self.documents]

        # Number of documents
        self.n_documents = len(abstracts)

        # Make vectorizer
        self.term_vectorizer = CountVectorizer(
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            ngram_range=(1, 1),
            analyzer="word",
        )

        print("\t{:8.2f}s: Vectorizing documents.".format(time() - self._start_time))

        # Vectorize documents
        self.matrix_doc_term = self.term_vectorizer.fit_transform(abstracts)

        # Compute document lengths
        self.document_lengths = self.matrix_doc_term.sum(1)
        self._avg_document_length = self.document_lengths.mean()

        print("\t{:8.2f}s: Computing TF-IDF.".format(time() - self._start_time))

        # Make TF-IDF transformer
        tfidf_transformer = TfidfTransformer(
            norm="l2",
            smooth_idf=True,
        )

        # Transform to TF-IDF
        _ = tfidf_transformer.fit_transform(self.matrix_doc_term)

        # Get IDF
        self.idf = tfidf_transformer.idf_

        # Number of features
        self.n_words = self.matrix_doc_term.shape[1]

    @staticmethod
    def _request_wikipedia_path():
        prompt = "\nBuilding prerequisites from Wikipedia dump." + \
                 "\nType path to Wikipedia .xml file. " + \
                 "\ndefault: '{}', 'quit' for stopping. ".format(WikipediaSearcher.__default_wikipedia_path)
        wiki_data_dir = input(prompt)
        if not wiki_data_dir:
            wiki_data_dir = WikipediaSearcher.__default_wikipedia_path.resolve()
        else:
            if "quit" == wiki_data_dir.lower().strip():
                return None
            wiki_data_dir = Path(wiki_data_dir).resolve()
        return wiki_data_dir

    @staticmethod
    def _use_prepared(always_load):
        use_preparsed = False
        if WikipediaSearcher.__parsed_documents_path.exists():
            if always_load:
                return True
            prompt = "\nParsed documents found in {}".format(WikipediaSearcher.__parsed_documents_path) + \
                     "\nDo you want to use this file? (Y/n) "
            answer = input(prompt)
            if not answer or "y" in answer.lower():
                use_preparsed = True
        return use_preparsed

    @staticmethod
    def _use_processed(always_load):
        use_processed = False
        if WikipediaSearcher.__processed_documents_path.exists():
            if always_load:
                return True
            prompt = "\nProcessed documents found in {}\n".format(WikipediaSearcher.__processed_documents_path) + \
                     "Do you want to use this file? (Y/n) "
            answer = input(prompt)
            if not answer or "y" in answer.lower():
                use_processed = True
        return use_processed

    @staticmethod
    def _process_xml_element(element: Element, fields_of_interest):
        output = WikipediaDocument()
        for child in element:
            for name in fields_of_interest:
                if name in child.tag:
                    output.__setattr__(name, child.text)

        return output

    def _parse_wikipedia(self, out_start_tag="<doc>", out_end_tag="</doc>",
                         fields_of_interest=("title", "abstract", "url"), max_lines=None):
        assert self._wiki_data_dir.exists(), "Wikipedia data does not exist"

        # Start
        print("\nStarting parsing of Wikiepdia file: {}".format(self._wiki_data_dir))

        # Number of lines in file
        print("\t{:8.2f}s: Counting lines in XML-file: ".format(time() - self._start_time))
        n_lines = raw_buffered_line_counter(self._wiki_data_dir)
        print("\t\t\t\t\t{:,d} lines.".format(n_lines))

        # Open file
        print("\tProcessing file:")
        with self._wiki_data_dir.open("r") as file:

            # Initialize
            buffer = []
            append = False
            documents = []

            # Go through lines
            for line_nr, line in enumerate(file):

                if line_nr % (int(n_lines / 100)) == 0:
                    print("\t{:8.2f}s: {} / {} ({:.2%})"
                          .format(time() - self._start_time, line_nr, n_lines, line_nr / n_lines))

                # Check for hard-break
                if isinstance(max_lines, int) and line_nr > max_lines:
                    break

                if out_start_tag in line:
                    buffer = []
                    append = True

                if append:
                    buffer.append(line)

                if out_end_tag in line:
                    append = False

                    # Process xml-element
                    element = fromstring("".join(buffer))
                    documents.append(self._process_xml_element(element=element, fields_of_interest=fields_of_interest))

                    buffer = []

        # Store
        self.documents = documents

    @property
    def vocabulary(self):
        return self.term_vectorizer.vocabulary_

    def __str__(self):
        return "{}({} documents, {} words)".format(type(self).__name__, self.n_documents, self.n_words)

    def __repr__(self):
        return str(self)

    def search(self, search, k=1.5, b=0.75, search_min_threshold=0, max_results=None):
        # Vectorize search
        m_search_term = self.term_vectorizer.transform([search])
        m_search_bool = (m_search_term > 0).toarray()[0, :]

        # Get term-frequencies of query for all documents
        doc_f = self.matrix_doc_term[:, m_search_bool]

        # Get IDF of the search words
        idf_search = self.idf[m_search_bool]

        # Compute okapi bm25 score
        numerator = doc_f * (k + 1)
        denominator = doc_f + k * (1 - b + b * (self.document_lengths / self._avg_document_length))
        fraction = numerator / denominator
        scores = np.squeeze(np.array(fraction * np.expand_dims(idf_search, axis=1)), axis=1)

        # Sort search results
        sort_locs = np.argsort(a=-scores, kind="quicksort")
        sort_scores = scores[sort_locs]

        # Remove zero-scores
        thresh_locs = sort_scores > search_min_threshold
        sort_locs = sort_locs[thresh_locs]
        sort_scores = sort_scores[thresh_locs]

        # Weave indices and scores for output
        output = [(idx, c_score) for idx, c_score in zip(sort_locs, sort_scores)]

        # Restrict number of outputs
        if isinstance(max_results, int):
            output = output[:max_results]

        return output


if __name__ == "__main__":
    searcher = WikipediaSearcher()
