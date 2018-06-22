import gzip
import sys
from pathlib import Path
from xml.etree.cElementTree import Element, fromstring

import numpy
import pandas
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm, tqdm_notebook

from src.utility.connectivity import retrieve_file
from src.utility.files import (
    ensure_directory,
    save_as_compressed_pickle_file,
    load_from_compressed_pickle_file
)

if not sys.stdout.isatty():
    tqdm = tqdm_notebook

_data_dir = Path("data", "wikipedia")
ensure_directory(_data_dir)

WIKIPEDIA_URL = "https://dumps.wikimedia.org/enwiki/latest/"\
                "enwiki-latest-abstract.xml.gz"

WIKIPEDIA_FILENAME = WIKIPEDIA_URL.split("/")[-1]
WIKIPEDIA_BASE_NAME = WIKIPEDIA_FILENAME.split(".")[0]

WIKIPEDIA_DOCUMENT_TITLE_PREFIX = "Wikipedia: "

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


class Wikipedia:
    __default_wikipedia_filename = WIKIPEDIA_FILENAME
    __parsed_documents_path = Path(_data_dir,
        f"{WIKIPEDIA_BASE_NAME}-parsed.pkl.gz")
    __vectorised_documents_path = Path(_data_dir,
        f"{WIKIPEDIA_BASE_NAME}-vectorised.pkl.gz")
    __fields_of_interest = ["title", "abstract", "url"]
    __out_start_tag = "<doc>"
    __out_end_tag = "</doc>"

    def __init__(self, max_lines=None, always_load=False):
        self.documents = None  # type: list[WikipediaDocument]

        # Check whether vectorised documents can be used
        use_vectorised = self._use_vectorised(always_load=always_load)

        # If we have the vectorised storage then load
        if use_vectorised:
            print("Loading preprocessed documents.")
            self._vectorised_storage = load_from_compressed_pickle_file(
                Wikipedia.__vectorised_documents_path)

        # Check whether parsed documents can be used
        use_parsed = self._use_parsed(always_load=always_load)

        # If parsed documents are to be used, then get these
        if use_parsed:
            print("Loading parsed documents.")
            self.documents = load_from_compressed_pickle_file(
                Wikipedia.__parsed_documents_path)

        # Otherwise start processing a Wikipedia file
        else:

            # Get wikipedia data path
            self._wikipedia_data_directory = \
                self._request_wikipedia_directory()
            self._wikipedia_data_path = Path(
                self._wikipedia_data_directory,
                Wikipedia.__default_wikipedia_filename
            )
            if self._wikipedia_data_path is None:
                return

            # Download Wikipedia file
            if not self._wikipedia_data_path.exists():
                print("Downloading Wikipedia documents.")
                self._download_wikipedia()

            # Parse Wikipedia file
            print("Parsing Wikipedia documents.")
            self._parse_wikipedia(max_lines=max_lines)

            # Store data
            print("Saving parsed documents.")
            save_as_compressed_pickle_file(
                self.documents,
                Wikipedia.__parsed_documents_path,
            )

            # Process documents
            self._process_parsed_documents()

            # Store vectorised data
            print("Saving preprocessed documents.")
            save_as_compressed_pickle_file(
                self._vectorised_storage,
                Wikipedia.__vectorised_documents_path.open("wb"),
            )

        # Progress
        print("\nWikipedia loaded.")

    @property
    def _vectorised_storage(self):
        return (self.n_documents,
                self.n_words,
                self.matrix_doc_term,
                self.document_lengths,
                self._avg_document_length,
                self.idf,
                self.term_vectorizer)

    @_vectorised_storage.setter
    def _vectorised_storage(self, values):
        (self.n_documents, self.n_words, self.matrix_doc_term,
         self.document_lengths, self._avg_document_length,
         self.idf, self.term_vectorizer) = values

    def _process_parsed_documents(self):

        print("Preprocessing documents.")

        # Get titles and abstracts
        documents = []
        for document in self.documents:

            if document.abstract:
                abstract = document.abstract
            else:
                abstract = ""

            if document.title:
                title = document.title
                if title.startswith(WIKIPEDIA_DOCUMENT_TITLE_PREFIX):
                    title = title.replace(
                        WIKIPEDIA_DOCUMENT_TITLE_PREFIX,
                        "",
                        1
                    )
            else:
                title = ""

            if title: # and title not in abstract:
                title_and_abstract = "\n\n".join([title, abstract])
            else:
                title_and_abstract = abstract

            documents.append(title_and_abstract)

        # Number of documents
        self.n_documents = len(documents)

        # Make vectorizer
        self.term_vectorizer = CountVectorizer(
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            ngram_range=(1, 1),
            analyzer="word",
        )

        # Vectorize documents
        self.matrix_doc_term = self.term_vectorizer.fit_transform(documents)

        # Compute document lengths
        self.document_lengths = self.matrix_doc_term.sum(1)
        self._avg_document_length = self.document_lengths.mean()

        print("Computing TF-IDF.")

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
    def _request_wikipedia_directory():
        prompt = "Path to Wikipedia directory "\
                 "(default is '{}'; type 'Cancel' to abort):"\
                 .format(_data_dir)
        wiki_data_dir = input(prompt)
        if not wiki_data_dir:
            wiki_data_dir = _data_dir.resolve()
        else:
            if wiki_data_dir.lower().strip() in ["quit", "cancel"]:
                return None
            wiki_data_dir = Path(wiki_data_dir).resolve()
        print(wiki_data_dir)
        return wiki_data_dir

    @staticmethod
    def _use_parsed(always_load):
        use_parsed = False
        if Wikipedia.__parsed_documents_path.exists():
            if always_load:
                return True
            prompt = "\nParsed documents found in {}".format(Wikipedia.__parsed_documents_path) + \
                     "\nDo you want to use this file? (Y/n) "
            answer = input(prompt)
            if not answer or answer.lower().strip() in ["y", "yes"]:
                use_parsed = True
        return use_parsed

    @staticmethod
    def _use_vectorised(always_load):
        use_vectorised = False
        if Wikipedia.__vectorised_documents_path.exists():
            if always_load:
                return True
            prompt = "\nPreprocessed documents found in {}\n".format(Wikipedia.__vectorised_documents_path) + \
                     "Do you want to use this file? (Y/n) "
            answer = input(prompt)
            if not answer or answer.lower().strip() in ["y", "yes"]:
                use_vectorised = True
        return use_vectorised

    @staticmethod
    def _process_xml_element(element: Element, fields_of_interest):
        output = WikipediaDocument()
        for child in element:
            for name in fields_of_interest:
                if name in child.tag:
                    output.__setattr__(name, child.text)

        return output

    def _download_wikipedia(self):
        retrieve_file(
            url=WIKIPEDIA_URL,
            path=self._wikipedia_data_path,
            title="Downloading"
        )

    def _parse_wikipedia(self, out_start_tag="<doc>", out_end_tag="</doc>",
                         fields_of_interest=("title", "abstract", "url"),
                         max_lines=None):
        assert self._wikipedia_data_path.exists(), "Wikipedia data does not exist"

        # Determine size of file
        compressed_size = self._wikipedia_data_path.stat().st_size

        # Open file

        progress_bar = tqdm(desc="Parsing", total=compressed_size,
                            unit="B", unit_scale=True)

        with self._wikipedia_data_path.open(mode="rb") as compressed_file:
            with gzip.GzipFile(fileobj=compressed_file) as uncompressed_file:

                # Initialize
                buffer = []
                append = False
                documents = []
                total_compressed_bytes_read_at_last_line = 0

                # Go through lines
                for line_number, line in enumerate(uncompressed_file):

                    line = line.decode("utf-8")

                    # Check for hard-break
                    if isinstance(max_lines, int) and line_number > max_lines:
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
                        documents.append(
                            self._process_xml_element(
                                element=element,
                                fields_of_interest=fields_of_interest
                            )
                        )

                        buffer = []

                    total_compressed_bytes_read = compressed_file.tell()
                    compressed_bytes_read_for_line = \
                        total_compressed_bytes_read \
                        - total_compressed_bytes_read_at_last_line
                    total_compressed_bytes_read_at_last_line = \
                        total_compressed_bytes_read
                    progress_bar.update(compressed_bytes_read_for_line)

                progress_bar.close()

        # Store
        self.documents = documents

    @property
    def vocabulary(self):
        return self.term_vectorizer.vocabulary_

    def __str__(self):
        return "{}({} documents, {} words)".format(type(self).__name__, self.n_documents, self.n_words)

    def __repr__(self):
        return str(self)

    def search(self, query, k_1=1.5, b=0.75):

        # Vectorise query and get keyword indices
        query_vectorised = self.term_vectorizer.transform([query])
        _, keyword_indices = query_vectorised.nonzero()

        # Term frequencies of keywords for all documents
        tf_matrix = self.matrix_doc_term[:, keyword_indices]

        # IDF of keywords
        idf_vector = numpy.expand_dims(self.idf[keyword_indices], axis=1)

        # Normalised document lengths
        l = self.document_lengths / self._avg_document_length

        # Nonzero indices for term-frequency matrix for sparse optimisations
        nonzero_rows, nonzero_columns = tf_matrix.nonzero()

        # For the additional term in the denominator of the fraction in the
        # Okapi score equation, only entries which will be nonzero in the
        # resulting matrix from the division are computed, and this matrix is
        # then encoded as a sparse matrix to take advantage of sparse
        # operations
        denominator_addend_matrix_data = k_1 * (1 - b + b * l[nonzero_rows])
        denominator_addend_matrix = scipy.sparse.csr_matrix(
            (
                denominator_addend_matrix_data.A.flatten(),
                (nonzero_rows, nonzero_columns)
            ),
            tf_matrix.shape
        )

        # Compute fraction Okapi score equation
        numerator_matrix = tf_matrix * (k_1 + 1)
        denominator_matrix = tf_matrix + denominator_addend_matrix
        fraction_matrix = numerator_matrix / denominator_matrix
        fraction_matrix = scipy.sparse.csr_matrix(
            (
                fraction_matrix[nonzero_rows, nonzero_columns].A.flatten(),
                (nonzero_rows, nonzero_columns)
            ),
            shape=tf_matrix.shape
        )

        # Compute Okapi BM25 scores
        scores = fraction_matrix @ idf_vector

        # Package scores
        scores = pandas.Series(scores.flatten())
        
        # Remove zero-scores
        scores = scores[scores > 0]
        
        # Sort scores
        scores = scores.sort_values(
            ascending=False,
            kind="quicksort"
        )

        return scores


if __name__ == "__main__":
    wikipedia = Wikipedia()
