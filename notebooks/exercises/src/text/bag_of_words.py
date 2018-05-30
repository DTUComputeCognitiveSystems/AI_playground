import numpy
import pandas
import seaborn
# import scipy.sparse
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


COLOUR_MAP = seaborn.cubehelix_palette(as_cmap=True)


class BagOfWords:
    def __init__(self, corpus):

        if not isinstance(corpus, list):
            corpus = [corpus]

        self.__corpus = corpus
        self.__count_matrix = self.__tfidf_matrix = self.__vocabulary = None
        self.__number_of_documents = self.__number_of_words = None

        self.update(self.__corpus)

    @property
    def corpus(self):
        return self.__corpus

    @property
    def count_matrix(self):
        return self.__count_matrix

    @property
    def count_data_frame(self):
        return self.__as_data_frame(tfidf=False)

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def tfidf_data_frame(self):
        return self.__as_data_frame(tfidf=True)

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def number_of_documents(self):
        return self.__number_of_documents

    @property
    def number_of_words(self):
        return self.__number_of_words

    def __as_data_frame(self, tfidf=False):

        if tfidf:
            matrix = self.__tfidf_matrix
        else:
            matrix = self.__count_matrix

        document_number = [i + 1 for i in range(self.__number_of_documents)]

        data_frame = pandas.DataFrame(
            matrix.toarray(),
            index=document_number,
            columns=self.__vocabulary
        )

        return data_frame

    def update(self, corpus):

        count_vectoriser = CountVectorizer()
        tfidf_transformer = TfidfTransformer()

        self.__count_matrix = count_vectoriser.fit_transform(corpus)
        self.__tfidf_matrix = tfidf_transformer.fit_transform(
            self.__count_matrix)
        self.__vocabulary = count_vectoriser.get_feature_names()

        self.__number_of_documents, self.__number_of_words = \
            self.__count_matrix.shape

    def plot(self, tfidf=False):

        data_frame = self.__as_data_frame(tfidf=tfidf)

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
                minimum_value, maximum_value, maximum_value + 1)

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
