import numpy
import pandas
import seaborn
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


COLOUR_MAP = seaborn.cubehelix_palette(as_cmap=True)
MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES = 25


class BagOfWords:
    def __init__(self, corpus, vocabulary=None, stop_words=None):

        self.__corpus = ensure_list_input(corpus)
        self.__vocabulary = ensure_vocabulary_input(vocabulary)
        self.__stop_words = ensure_list_input(stop_words)

        self.__count_matrix = self.__tfidf_matrix = self.__words = None
        self.__number_of_documents = self.__number_of_words = None

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
    def number_of_documents(self):
        return self.__number_of_documents

    @property
    def number_of_words(self):
        return self.__number_of_words

    def __as_data_frame(self, tfidf=False, remove_absent_words=False):

        if tfidf:
            matrix = self.__tfidf_matrix
        else:
            matrix = self.__count_matrix

        document_numbers = [i + 1 for i in range(self.__number_of_documents)]

        data_frame = pandas.SparseDataFrame(
            matrix,
            index=document_numbers,
            columns=self.__words,
            default_fill_value=0
        ).astype(matrix.dtype)

        if remove_absent_words:
            word_sums = data_frame.sum(axis="index")
            absent_words = word_sums[word_sums == 0].index
            data_frame = data_frame.drop(columns=absent_words)

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

        self.__number_of_documents, self.__number_of_words = \
            self.__count_matrix.shape

    def filter_vocabulary(self, filters=[]):

        stop_words = []

        for word in self.__vocabulary:
            for filter_ in filters:
                if filter_(word):
                    stop_words.append(word)

        if self.__stop_words:
            self.__stop_words += stop_words
        else:
            self.__stop_words = stop_words

        self.__update()

    def plot(self, tfidf=False, remove_absent_words=None):

        if remove_absent_words is None:
            if self.number_of_words \
                    > MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES:
                remove_absent_words = True
            else:
                remove_absent_words = False

        data_frame = self.__as_data_frame(
            tfidf=tfidf,
            remove_absent_words=remove_absent_words
        ).to_dense()

        if self.__number_of_documents <= 10 and self.__number_of_words <= 20:
            annotate = True
            line_width = 1
            y_tick_labels = "auto"
            colour_bar = False
            colour_bar_parameters = {}
        else:
            annotate = False
            line_width = 0
            y_tick_labels = max(self.__number_of_documents // 4, 5)
            colour_bar = True
            colour_bar_parameters = {
                "orientation": "horizontal"
            }

        if tfidf:
            annotation_format = ".2g"
        else:
            annotation_format = "d"

            minimum_value = data_frame.values.min()
            maximum_value = data_frame.values.max()

            colour_bar_ticks = numpy.linspace(
                minimum_value, maximum_value, maximum_value + 2)

            colour_bar_parameters["boundaries"] = colour_bar_ticks
            colour_bar_parameters["ticks"] = colour_bar_ticks

        figure = pyplot.figure(figsize=(10, 4), dpi=150)

        if colour_bar:
            grid_spec_parameters = {"height_ratios": (.9, .05), "hspace": .3}
            axis, colour_bar_axis = figure.subplots(
                nrows=2,
                gridspec_kw=grid_spec_parameters
            )
        else:
            axis = figure.add_subplot(1, 1, 1)
            colour_bar_axis = None

        seaborn.heatmap(
            data_frame.reindex(index=data_frame.index[::-1]),
            yticklabels=y_tick_labels,
            cmap=COLOUR_MAP,
            annot=annotate,
            fmt=annotation_format,
            square=True,
            linewidths=line_width,
            ax=axis,
            cbar=colour_bar,
            cbar_ax=colour_bar_axis,
            cbar_kws=colour_bar_parameters
        )
        axis.invert_yaxis()

        x_tick_labels = axis.get_xticklabels()
        y_tick_labels = axis.get_yticklabels()

        if x_tick_labels[0].get_rotation() == 90.0:
            axis.set_xticklabels(
                axis.get_xticklabels(),
                horizontalalignment="right",
                rotation_mode="anchor",
                rotation=45
            )

        if y_tick_labels[0].get_rotation() == 90.0:
            axis.set_yticklabels(
                axis.get_yticklabels(),
                rotation="horizontal"
            )


def ensure_list_input(value):
    if value and not isinstance(value, list):
        value = [value]
    return value


def ensure_vocabulary_input(value):
    if not isinstance(value, (dict, set)):
        value = ensure_list_input(value)
    return value
