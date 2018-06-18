import numpy
import pandas
import scipy.sparse
import seaborn
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordcloud import WordCloud

from src.text.bag_of_words import bag_of_words_utilities


seaborn.set(
    context="notebook",
    style="ticks",
    palette="Set2",
    rc={"lines.markersize": 5}
)


class BagOfWords:
    def __init__(self, corpus, vocabulary=None, stop_words=None):

        self.__corpus = ensure_list_input(corpus)
        self.__vocabulary = ensure_vocabulary_input(vocabulary)
        self.__stop_words = ensure_list_input(stop_words)

        self.__count_matrix = self.__tfidf_matrix = None
        self.__words = self.__idf = None

        self.__update()

    @property
    def corpus(self):
        return self.__corpus

    @corpus.setter
    def corpus(self, value):
        self.__corpus = ensure_list_input(value)
        self.__update()

    @property
    def vocabulary(self):
        return self.__vocabulary

    @vocabulary.setter
    def vocabulary(self, value):
        self.__vocabulary = ensure_vocabulary_input(value)
        self.__update()

    @property
    def stop_words(self):
        return self.__stop_words

    @stop_words.setter
    def stop_words(self, value):
        self.__stop_words = ensure_list_input(value)
        self.__update()

    @property
    def count_matrix(self):
        return self.__count_matrix

    @property
    def count_table(self):
        return self.__as_data_frame(tfidf=False, remove_absent_words=True)

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def tfidf_table(self):
        return self.__as_data_frame(tfidf=True, remove_absent_words=True)

    @property
    def words(self):
        return self.__words

    @property
    def idf(self):
        return self.__idf

    @property
    def document_numbers(self):
        return [i + 1 for i in range(self.number_of_documents)]

    @property
    def number_of_documents(self):
        return self.count_matrix.shape[0]

    @property
    def number_of_words(self):
        return self.count_matrix.shape[1]

    def __as_data_frame(self, tfidf=False, remove_absent_words=False):

        if tfidf:
            matrix = self.tfidf_matrix
        else:
            matrix = self.count_matrix
        
        data_frame = bag_of_words_utilities.bag_of_words_as_data_frame(
            matrix=matrix,
            terms=self.words,
            document_ids=self.document_numbers,
            remove_absent_terms=remove_absent_words
        )

        return data_frame

    def __update(self):

        count_vectoriser = CountVectorizer(
            encoding="utf-8",
            strip_accents=None,
            analyzer="word",
            stop_words=self.__stop_words,
            lowercase=True,
            max_df=1.0,
            min_df=0.0,
            max_features=None,
            vocabulary=self.__vocabulary,
            binary=None
        )
        tfidf_transformer = TfidfTransformer()

        # Ensure 1-character words are counted if present in vocabulary
        if self.__vocabulary and min(map(len, self.__vocabulary)) == 1:
            count_vectoriser.token_pattern = r"(?u)\b\w+\b"

        self.__count_matrix = count_vectoriser.fit_transform(self.__corpus)
        self.__tfidf_matrix = tfidf_transformer.fit_transform(
            self.__count_matrix)
        self.__vocabulary = count_vectoriser.vocabulary_
        self.__words = count_vectoriser.get_feature_names()
        self.__idf = tfidf_transformer.idf_

    # def filter_vocabulary(self, filters=[]):
    #
    #     stop_words = []
    #
    #     for word in self.__vocabulary:
    #         for filter_ in filters:
    #             if filter_(word):
    #                 stop_words.append(word)
    #
    #     if self.__stop_words:
    #         self.__stop_words += stop_words
    #     else:
    #         self.__stop_words = stop_words
    #
    #     self.__update()

    def plot_heat_map(self, kind="count", remove_absent_words=None):

        kind = bag_of_words_utilities.ensure_kind_of_bag_of_words(kind)

        if kind == "count":
            matrix = self.__count_matrix
        elif kind == "tfidf":
            matrix = self.tfidf_matrix

        bag_of_words_utilities.plot_heat_map_of_bag_of_words(
            bag_of_words=matrix,
            kind=kind,
            terms=self.words,
            document_ids=self.document_numbers,
            corpus=self.corpus,
            remove_absent_terms=remove_absent_words
        )

    def plot_word_cloud(self, kind="count"):

        idf = None

        if kind.lower() == "idf":
            idf=self.idf
            kind="count"

        kind = bag_of_words_utilities.ensure_kind_of_bag_of_words(kind)

        if kind == "count":
            matrix = self.__count_matrix
        elif kind == "tfidf":
            matrix = self.tfidf_matrix

        bag_of_words_utilities.plot_term_cloud_for_bag_of_words(
            bag_of_words_matrix=matrix,
            idf=idf,
            terms=self.words
        )

    def plot_pca(self, kind="count"):

        kind = bag_of_words_utilities.ensure_kind_of_bag_of_words(kind)

        if kind == "count":
            matrix = self.__count_matrix
        elif kind == "tfidf":
            matrix = self.tfidf_matrix

        bag_of_words_utilities.plot_pca_of_bag_of_words(
            bag_of_words_matrix=matrix,
            kind=kind,
            document_ids=self.document_numbers,
            corpus=self.corpus
        )


def ensure_list_input(value):
    if value and not isinstance(value, list):
        value = [value]
    return value


def ensure_vocabulary_input(value):
    if not isinstance(value, (dict, set)):
        value = ensure_list_input(value)
    return value
