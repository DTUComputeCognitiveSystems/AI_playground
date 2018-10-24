import bz2
import re
import sys
from collections import namedtuple
from pathlib import Path
from xml.etree import cElementTree as ElementTree
from urllib.parse import urljoin
from urllib.error import URLError

import mwparserfromhell
import numpy
import pycountry
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm, tqdm_notebook

from src.text.bag_of_words.okapi_bm25_search import OkapiBM25Searcher
from src.utility.connectivity import retrieve_file
from src.utility.files import (
    ensure_directory,
    save_as_compressed_pickle_file,
    load_from_compressed_pickle_file
)

if not sys.stdout.isatty():
    tqdm = tqdm_notebook

DATA_DIRECTORY = Path("data", "wikipedia")
ensure_directory(DATA_DIRECTORY)

WIKIPEDIA_PAGE_BASE_URL = "https://{}.wikipedia.org/wiki/"

WIKIPEDIA_DUMP_URL = "https://dumps.wikimedia.org/"\
    "{0}wiki/latest/"\
    "{0}wiki-latest-pages-articles-multistream.xml.bz2"

CC_BY_SA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
GNU_FREE_DOCUMENTATION_LICENSE_URL = "https://www.gnu.org/copyleft/fdl.html"

LICENSE_URLS = [
    CC_BY_SA_LICENSE_URL,
    GNU_FREE_DOCUMENTATION_LICENSE_URL
]

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

    def __init__(self,
                 language="English",
                 cache_directory_url="tmp",
                 maximum_number_of_documents=None):

        self.documents = None  # type: list[WikipediaDocument]

        self.__language_code = pycountry.languages.lookup(language).alpha_2
        self.__maximum_number_of_documents = maximum_number_of_documents

        # Wikipedia URLs
        self.__page_base_url = WIKIPEDIA_PAGE_BASE_URL.format(
            self.__language_code)
        self.__dump_url = WIKIPEDIA_DUMP_URL.format(self.__language_code)

        # Local files

        self.__filename = self.__dump_url.split("/")[-1]
        self.__path = Path(
            DATA_DIRECTORY,
            self.__filename
        )

        base_name = self.__filename.split(".")[0]

        self.__parsed_documents_filename = \
            base_name + "-parsed.pkl.gz"
        self.__parsed_documents_path = Path(
            DATA_DIRECTORY, self.__parsed_documents_filename)

        self.__vectorised_documents_filename = \
            base_name + "-vectorised.pkl.gz"
        self.__vectorised_documents_path = Path(
            DATA_DIRECTORY, self.__vectorised_documents_filename)

        # Load, download cache, or parse and preprocess as necessary

        if self.__parsed_documents_path.exists():
            self._load_parsed_documents()
            if self.__vectorised_documents_path.exists():
                self._load_vectorised_documents()
            else:
                self._vectorise_documents()

        else:
            parsed_documents_loaded_succesfully = False

            if cache_directory_url:

                parsed_documents_downloaded_succesfully = False
                vectorised_documents_downloaded_succesfully = False

                try:
                    print("Downloading parsed Wikipedia documents.")
                    parsed_documents_url = urljoin(
                        cache_directory_url,
                        self.__parsed_documents_filename
                    )
                    retrieve_file(
                        url=parsed_documents_url,
                        path=self.__parsed_documents_path
                    )
                    parsed_documents_downloaded_succesfully = True

                except URLError as url_error:
                    print(f"Failed to download documents ({url_error}).")
                    print("Falling back to parsing Wikipedia locally.")
                
                if parsed_documents_downloaded_succesfully:
                    self._load_parsed_documents()
                    parsed_documents_loaded_succesfully = True

                    try:
                        print("Downloading preprocessed Wikipedia documents.")
                        vectorised_documents_url = urljoin(
                            cache_directory_url,
                            self.__vectorised_documents_filename
                        )
                        retrieve_file(
                            url=vectorised_documents_url,
                            path=self.__vectorised_documents_path
                        )
                        vectorised_documents_downloaded_succesfully = True
                    except URLError as url_error:
                        print(f"Failed to download documents ({url_error}).")
                        print("Falling back to preprocess Wikipedia locally.")

                    if vectorised_documents_downloaded_succesfully:
                        self._load_vectorised_documents()
                    else:
                        self._vectorise_documents()

            if not parsed_documents_loaded_succesfully:
                if not self.__path.exists():
                    print("Downloading Wikipedia documents.")
                    self._download_wikipedia()
                self._parse_documents()
                self._vectorise_documents()

        # Okapi BM25 searcher
        self.searcher = OkapiBM25Searcher(
            tf_matrix=self.matrix_doc_term,
            idf_vector=self.idf
        )

        # Progress
        print("Wikipedia loaded.")

    @property
    def language_code(self):
        return self.__language_code

    def _load_parsed_documents(self):
        print("Loading parsed documents.")
        parsed_documents = load_from_compressed_pickle_file(
            self.__parsed_documents_path)
        self.__set_documents(parsed_documents["content"])

    def _load_vectorised_documents(self):
        print("Loading preprocessed documents.")
        vectorised_storage = load_from_compressed_pickle_file(
            self.__vectorised_documents_path)
        self._vectorised_storage = vectorised_storage["content"]

    def _parse_documents(self):

        # Parse Wikipedia file
        print("Parsing Wikipedia documents.")
        documents = self._parse_wikipedia(
            maximum_number_of_documents=self.__maximum_number_of_documents
        )

        # Store data
        print("Saving parsed documents.")
        parsed_documents = {
            "content": documents,
            "changes": "extracted first paragraph of each article",
            "license": LICENSE_URLS
        }
        save_as_compressed_pickle_file(
            parsed_documents,
            self.__parsed_documents_path,
        )

        self.__set_documents(documents)

    def __set_documents(self, documents):
        self.documents = []

        for document in documents:
            self.documents.append(
                WikipediaDocument(
                    title=document["title"],
                    url=document["url"],
                    abstract=document["abstract"]
                )
            )

    def _vectorise_documents(self):

        # Process documents
        self._process_parsed_documents()

        # Store vectorised data
        print("Saving preprocessed documents.")
        vectorised_storage = {
            "content": self._vectorised_storage,
            "changes": "bag-of-words representation of the first "
                       "paragraph of each article",
            "license": LICENSE_URLS
        }
        save_as_compressed_pickle_file(
            vectorised_storage,
            self.__vectorised_documents_path.open("wb"),
        )

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

        # Get abstracts
        documents = [document.abstract for document in self.documents]

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
        tfidf_transformer.fit_transform(self.matrix_doc_term)

        # Get IDF
        self.idf = tfidf_transformer.idf_

        # Number of features
        self.n_words = self.matrix_doc_term.shape[1]

    def _download_wikipedia(self):
        retrieve_file(
            url=self.__dump_url,
            path=self.__path
        )

    def _parse_wikipedia(self, maximum_number_of_documents=None):
        assert self.__path.exists(), "Wikipedia data does not exist"

        # Determine size of file
        compressed_size = self.__path.stat().st_size

        # Initialise container for documents
        documents = []

        with open(self.__path, mode="rb") as compressed_file:
            with bz2.BZ2File(compressed_file, mode="rb") as uncompressed_file:

                total_compressed_bytes_read_at_last_batch = 0
                tag_prefix = ""
                namespaces = []
                article_namespace_key = None
                in_page = False

                with tqdm(desc="", total=compressed_size, unit="B",
                          unit_scale=True) as progress_bar:

                    for event_number, (event, element) in enumerate(
                            ElementTree.iterparse(
                                uncompressed_file,
                                events=["start", "end", "start-ns", "end-ns"]
                            )
                        ):

                        if event == "start-ns":
                            namespaces.append(element)
                            namespace_id, namespace_uri = element
                            if namespace_id == "":
                                tag_prefix = f"{{{namespace_uri}}}"

                        elif event == "end-ns":
                            namespace = namespaces.pop()
                            namespace_id, namespace_uri = namespace
                            if namespace_id == "":
                                tag_prefix = ""

                        elif event == "start":
                            if element.tag == f"{tag_prefix}page":
                                in_page = True
                                title = None
                                text = None
                                page_namespace_keys = []
                                page_redirect = False

                        elif event == "end":

                            tag = element.tag

                            if tag.startswith(tag_prefix):
                                tag = tag.replace(tag_prefix, "", 1)

                            if tag == "namespace":
                                if element.text is None:
                                    article_namespace_key = element.attrib["key"]

                            elif in_page and tag == "title":
                                if not title:
                                    title = element.text
                                else:
                                    progress_bar.write(
                                        "Multiple titles found for article "
                                        f"\"{title}\". First one used."
                                    )

                            elif in_page and tag == "text":
                                if not text:
                                    text = element.text
                                else:
                                    progress_bar.write(
                                        "Multiple text sections found for article "
                                        f"\"{title}\". First one used."
                                    )

                            elif in_page and tag == "ns":
                                page_namespace_keys.append(element.text)

                            elif in_page and tag == "redirect":
                                page_redirect = True

                            elif in_page and tag == "page":

                                in_page = False

                                if article_namespace_key not in page_namespace_keys \
                                    or page_redirect:
                                        continue

                                url = self.__page_base_url \
                                    + title.replace(" ", "_")

                                abstract = self._parse_wikipedia_article(
                                    article_text=text,
                                    sections="first paragraph",
                                    include_header_image_captions=False,
                                    include_header_infoboxes=False
                                )

                                fulltext = self._parse_wikipedia_article(
                                    article_text=text,
                                    sections="all",
                                    include_header_image_captions=False,
                                    include_header_infoboxes=False
                                )

                                document = {
                                    "title": title,
                                    "url": url,
                                    "abstract": abstract,
                                    "text": fulltext
                                }

                                documents.append(document)

                            element.clear()

                        if maximum_number_of_documents and \
                            len(documents) >= maximum_number_of_documents:
                                break

                        if event_number % 1000 == 0:
                            total_compressed_bytes_read = \
                                compressed_file.tell()
                            compressed_bytes_read_for_batch = \
                                total_compressed_bytes_read \
                                - total_compressed_bytes_read_at_last_batch
                            total_compressed_bytes_read_at_last_batch = \
                                total_compressed_bytes_read
                            progress_bar.update(
                                compressed_bytes_read_for_batch)

        return documents

    def _parse_wikipedia_article(self,
                                 article_text,
                                 sections="first paragraph",
                                 include_header_image_captions=False,
                                 include_header_infoboxes=False):
        text = ""
        if sections not in ["first paragraph", "lead", "all"]:
            raise ValueError(
                "Can only extract the first paragraph, the lead, or all sections."
            )

        def remove_footnotes_from_wikimedia_markup(markup):
            for element in markup.filter_tags():
                if element.tag == "ref":
                    try:
                        markup.remove(element)
                    except ValueError:
                        pass

        def add_line_break_before_lists_in_wikimedia_text(text):
            return re.sub(r"(?m)(^(\*|#) .+$)", r"\n\1", text)

        if sections == "all":
            article_text = add_line_break_before_lists_in_wikimedia_text(
                article_text)
            article_markup = mwparserfromhell.parse(article_text)
            sections_markup = article_markup.get_sections(
                include_lead = True, include_headings=True)
            lead_markup = sections_markup[0]
        else:
            # Remove everything after the first header including the header
            lead_text = re.split(r"==[^=]+==", article_text, maxsplit=1)[0]
            lead_text = add_line_break_before_lists_in_wikimedia_text(lead_text)
            lead_markup = mwparserfromhell.parse(lead_text)

        remove_footnotes_from_wikimedia_markup(lead_markup)

        header_infoboxes = []
        header_image_captions = []
        nodes_to_remove = []

        # Find the first line while saving image captions and info boxes as well
        # as removing the rest before the first line
        for node in lead_markup.nodes:

            if isinstance(node, mwparserfromhell.nodes.Wikilink) and node.text:
                nodes_to_remove.append(node)
                caption = node.text.strip_code().split("|")[-1]
                header_image_captions.append(caption)

            elif isinstance(node, mwparserfromhell.nodes.Template):
                nodes_to_remove.append(node)
                if "\n" in node:
                    infobox = {
                        parameter.name.strip_code().strip():
                        parameter.value.strip_code().strip()
                        for parameter in node.params
                    }
                    header_infoboxes.append(infobox)

            elif isinstance(node, mwparserfromhell.nodes.Comment):
                nodes_to_remove.append(node)

            elif isinstance(node, mwparserfromhell.nodes.Tag) \
                and node.tag == "table":
                    nodes_to_remove.append(node)

            elif isinstance(node, mwparserfromhell.nodes.Text):
                value = re.sub(r"__[A-Z]+__", "", node.value.lstrip())
                if not re.match(r"[A-Za-z0-9\"\']", value):
                    nodes_to_remove.append(node)
                else:
                    node.value = value
                    break

            else:
                break

        for node in nodes_to_remove:
            lead_markup.remove(node)

        lead = lead_markup.strip_code(
            normalize=True,
            collapse=True,
            keep_template_params=False
        )

        # Add line breaks before and after image links
        lead = re.sub(r"(?m)(^thumb\|.+$)", r"\n\1\n", lead)
        lead = re.sub(r"(?m)(^.+)(thumb\|.+$)", r"\1\n\n\2\n", lead)

        if sections == "first paragraph":
            text = lead.split("\n\n")[0]
            text = text.replace("\n", " ")
        elif sections == "lead":
            text = lead

        if include_header_image_captions and header_image_captions:
            text += "\n\n" + "\n\n".join(header_image_captions)

        if include_header_infoboxes and header_infoboxes:
            for infobox in header_infoboxes:
                infobox_string = "\n\n"
                for infobox_key, infobox_value in infobox.items():
                    infobox_string += f"* {infobox_key}: {infobox_value}\n"
                text += infobox_string

        if sections == "all":
            remaining_sections_markup = mwparserfromhell.parse(sections_markup[1:])
            #try:
            remove_footnotes_from_wikimedia_markup(
                    remaining_sections_markup)
            #except Exception:
                #print(sections_markup)

            text += remaining_sections_markup.strip_code(
                normalize=True,
                collapse=True,
                keep_template_params=False
            )

        text = re.sub(r"__[A-Z]+__\n?", "", text)
        text = text.replace("()", "")
        text = re.sub(r" +", " ", text)
        text = text.strip()

        return text

    @property
    def vocabulary(self):
        return self.term_vectorizer.vocabulary_

    @property
    def terms(self):
        return self.term_vectorizer.get_feature_names()

    def __str__(self):
        return "{}({} documents, {} words)".format(type(self).__name__, self.n_documents, self.n_words)

    def __repr__(self):
        return str(self)

    def search(self, query, k_1=1.5, b=0.75):

        query_vectorised = self.term_vectorizer.transform([query])

        scores = self.searcher.search(
            query_vectorised=query_vectorised,
            k_1=k_1,
            b=b
        )

        _, keyword_indices = query_vectorised.nonzero()
        keywords = numpy.array(self.terms)[keyword_indices].tolist()

        SearchResults = namedtuple("SearchResults", ["scores", "keywords"])
        results = SearchResults(scores=scores, keywords=keywords)

        return results

if __name__ == "__main__":
    wikipedia = Wikipedia()
