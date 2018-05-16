import codecs
from time import time
from xml.etree.cElementTree import Element, fromstring
from collections import Iterable
from pathlib import Path
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


###########################
# File processing

start_time = time()


def time_print(*args, **kwargs):
    print("{:8.2f}s:".format(time() - start_time), *args, **kwargs)


def raw_buffered_line_counter(path, encoding="utf-8", buffer_size=1024 * 1024):
    """
    Fast way to count the number of lines in a file.
    :param Path path: Path to file.
    :param str encoding: Encoding used in file.
    :param int buffer_size: Size of buffer for loading.
    :return: int
    """
    # Open file
    f = codecs.open(str(path), encoding=encoding, mode="r")

    # Reader generator
    def _reader_generator(reader):
        b = reader(buffer_size)
        while b:
            yield b
            b = reader(buffer_size)

    # Reader used
    file_read = f.raw.read

    # Count lines
    line_count = sum(buf.count(b'\n') for buf in _reader_generator(file_read)) + 1

    return line_count


file_path = Path("~", "Downloads", "enwiki-latest-abstract.xml").expanduser()

time_print("Lines in XML-file: ")
n_lines = raw_buffered_line_counter(file_path)
time_print("\t\t{:,d}".format(n_lines))

fields_of_interest = ["title", "abstract", "url"]
out_start_tag = "<doc>"
out_end_tag = "</doc>"


def process_element(element: Element):
    output = dict()
    for child in element:
        for name in fields_of_interest:
            if name in child.tag:
                output[name] = child.text

    return output


maxlines = None

# Open file
with file_path.open("r") as file:

    # Initialize
    buffer = []
    append = False
    docs = []

    # Go through lines
    for line_nr, line in enumerate(file):

        if line_nr % (int(n_lines / 100)) == 0:
            time_print("{} / {} ({:.2%})".format(line_nr, n_lines, line_nr / n_lines))

        # Check for hard-break
        if isinstance(maxlines, int) and line_nr > maxlines:
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
            docs.append(process_element(element))

            buffer = []

###########################
# Okapi preperations

# Get titles and abstracts
titles = [val[fields_of_interest[0]] if val[fields_of_interest[0]] is not None else "" for val in docs]
abstracts = [val[fields_of_interest[1]] if val[fields_of_interest[1]] is not None else "" for val in docs]
urls = [val[fields_of_interest[2]] if val[fields_of_interest[2]] is not None else "" for val in docs]

# Number of documents
n_docs = len(titles)

# Make vectorizer
vectorizer = CountVectorizer(
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    ngram_range=(1, 1),
    analyzer="word",
)

# Vectorize documents
time_print("Vectorizing documents")
m_doc_term = vectorizer.fit_transform(abstracts)

# Compute document lengths
doc_lengths = m_doc_term.sum(1)
avg_length = doc_lengths.mean()

# Make TF-IDF transformer
time_print("Document TFIDF")
tfidf_transformer = TfidfTransformer(
    norm="l2",
    smooth_idf=True,
)

# Transform to TF-IDF
m_doc_tfidf = tfidf_transformer.fit_transform(m_doc_term)

# Get IDF
idf = tfidf_transformer.idf_


###########################
# Search

time_print("Search")

search_min_threshold = 0

# Okapi variables
k = 1.5
b = 0.75

search = "when you wish upon a star"

# Vectorize search
m_search_term = vectorizer.transform([search])
m_search_bool = (m_search_term > 0).toarray()[0, :]

# Get term-frequencies of query for all documents
doc_f = m_doc_term[:, m_search_bool]

# Get IDF of the search words
idf_search = idf[m_search_bool]

# Compute okapi bm25 score
numerator = doc_f * (k + 1)
denominator = doc_f + k * (1 - b + b * (doc_lengths / avg_length))
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

time_print("Done!")
