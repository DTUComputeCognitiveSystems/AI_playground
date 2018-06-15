import numpy
import pandas
import scipy.sparse
import seaborn
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from wordcloud import WordCloud


COLOUR_MAP = seaborn.cubehelix_palette(as_cmap=True)
MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES = 25
MAXIMUM_NUMBER_OF_DOCUMENTS_FOR_SIMPLE_PLOT = 10
MAXIMUM_NUMBER_OF_WORDS_FOR_SIMPLE_PLOT = 25
MAXIMUM_VALUE_RANGE_FOR_DISCRETE_COLOUR_BAR = 25


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

        words = numpy.array(self.words)

        if remove_absent_words:
            word_sums = matrix.sum(axis=0).A.flatten()
            absent_word_indices = word_sums != 0
            matrix = matrix[:, absent_word_indices]
            words = words[absent_word_indices]

        data_frame = pandas.SparseDataFrame(
            matrix,
            index=self.document_numbers,
            columns=words.tolist(),
            default_fill_value=0
        )
        data_frame = data_frame.astype(matrix.dtype)

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

    def plot_heat_map(self, tfidf=False, remove_absent_words=None):

        # Data

        if remove_absent_words is None:
            if self.number_of_words \
                    > MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES:
                remove_absent_words = True
            else:
                remove_absent_words = False

        data_frame = self.__as_data_frame(
            tfidf=tfidf,
            remove_absent_words=remove_absent_words
        )
        data_frame = data_frame.reindex(index=data_frame.index[::-1])
        data_frame = data_frame.to_dense()

        # Setup

        if self.number_of_documents \
                <= MAXIMUM_NUMBER_OF_DOCUMENTS_FOR_SIMPLE_PLOT \
                and self.number_of_words \
                <= MAXIMUM_NUMBER_OF_WORDS_FOR_SIMPLE_PLOT:
            annotate = True
            line_width = 1
            y_tick_labels = "auto"
            colour_bar = False
            colour_bar_parameters = {}
        else:
            annotate = False
            line_width = 0
            y_tick_labels = max(self.number_of_documents // 4, 5)
            colour_bar = True
            colour_bar_parameters = {
                "orientation": "horizontal"
            }

        colour_bar_tick_labels = None

        if tfidf:
            value_kind = "tf-idf value"
            annotation_format = ".2g"
        else:
            value_kind = "Count"
            annotation_format = "d"

            minimum_value = data_frame.values.min()
            maximum_value = data_frame.values.max()
            value_range = maximum_value - minimum_value

            if value_range <= MAXIMUM_VALUE_RANGE_FOR_DISCRETE_COLOUR_BAR:

                colour_bar_boundaries = numpy.linspace(
                    minimum_value, maximum_value, value_range + 2)
                colour_bar_ticks = numpy.stack(
                    (colour_bar_boundaries[:-1], colour_bar_boundaries[1:])
                ).mean(axis=0)
                colour_bar_tick_labels = numpy.arange(
                    minimum_value, maximum_value + 1)

                colour_bar_parameters["boundaries"] = colour_bar_boundaries
                colour_bar_parameters["ticks"] = colour_bar_ticks

        # Plot

        figure = pyplot.figure(
            figsize=(10, 5),
            # dpi=150
        )

        if colour_bar:
            axis, colour_bar_axis = figure.subplots(
                nrows=2,
                gridspec_kw={
                    "height_ratios": (.9, .05),
                    "hspace": .3
                }
            )
        else:
            axis = figure.add_subplot(1, 1, 1)
            colour_bar_axis = None

        seaborn.despine()

        seaborn.heatmap(
            data_frame,
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

        axis.set_xlabel("Words")
        x_tick_labels = axis.get_xticklabels()

        if x_tick_labels[0].get_rotation() == 90.0:
            axis.set_xticklabels(
                axis.get_xticklabels(),
                horizontalalignment="right",
                rotation_mode="anchor",
                rotation=45
            )

        axis.set_ylabel("Document numbers")
        y_tick_labels = axis.get_yticklabels()

        if y_tick_labels[0].get_rotation() == 90.0:
            axis.set_yticklabels(
                axis.get_yticklabels(),
                rotation="horizontal"
            )

        if colour_bar_axis:
            colour_bar_axis.set_xlabel(value_kind)
            if colour_bar_tick_labels is not None:
                colour_bar_axis.set_xticklabels(colour_bar_tick_labels)

        # Hover annotation

        hover_annotation = axis.annotate(
            s="",
            xy=(0, 0),
            xytext=(0, 20),
            textcoords="offset points",
            bbox={
                "boxstyle": "square",
                "facecolor": "white"
            }
        )
        hover_annotation.set_visible(False)

        def update_hover_annotation(x, y):

            x, y = map(int, (x, y))

            hover_annotation.xy = (x, y)

            document_index = int(y)
            document_number = self.number_of_documents - document_index
            word_index = int(x)
            document = self.__corpus[document_number - 1]
            word = data_frame.columns[word_index]
            value = data_frame.values[document_index, word_index]

            text = "\n".join([
                f"Document number: {document_number}.",
                f"Document content: \"{document}\".",
                f"Word: \"{word}\".",
                f"{value_kind}: {value:{annotation_format}}."
            ])
            hover_annotation.set_text(text)

        def hover(event):
            if event.inaxes == axis:
                update_hover_annotation(x=event.xdata, y=event.ydata)
                hover_annotation.set_visible(True)
                figure.canvas.draw_idle()
            else:
                hover_annotation.set_visible(False)
                figure.canvas.draw_idle()

        figure.canvas.mpl_connect("motion_notify_event", hover)

    def plot_word_cloud(self, kind="count"):

        if kind in ["count", "tfidf"]:
            if kind == "count":
                matrix = self.__count_matrix
            elif kind == "tfidf":
                matrix = self.__tfidf_matrix
            sum_vector = scipy.sparse.coo_matrix(matrix.sum(axis=0))
            frequency_vector = sum_vector / sum_vector.max()
        elif kind == "idf":
            nonzero_indices = self.__count_matrix.sum(axis=0).nonzero()
            idf = self.__idf.copy()
            idf_normalised = (idf - idf.min()) / (idf.max() - idf.min())
            frequency_vector = scipy.sparse.coo_matrix((
                idf_normalised[nonzero_indices[1]], nonzero_indices
            ))
        else:
            raise ValueError("`kind` can only be 'count', 'tfidf', or 'idf'.")

        word_frequencies = {}

        for index, value in zip(frequency_vector.col,
                                frequency_vector.data):
            word = self.words[index]
            frequency = value
            word_frequencies[word] = frequency

        word_cloud_generator = WordCloud(
            width=400,
            height=200,
            prefer_horizontal=0.9,
            mask=None,
            scale=5,
            min_font_size=4,
            font_step=1,
            max_words=None,
            stopwords=None,
            background_color="white",
            mode="RGB",
            relative_scaling=0.5,
            regexp=None,
            collocations=True,
            colormap="rocket",
            normalize_plurals=False
        )
        word_cloud = word_cloud_generator.generate_from_frequencies(
            word_frequencies)

        figure = pyplot.figure(
            figsize=(10, 5),
            # dpi=150
            frameon=False
        )
        axis = pyplot.Axes(figure, [0., 0., 1., 1.])
        axis.set_axis_off()
        figure.add_axes(axis)

        axis.imshow(word_cloud, interpolation="bilinear", aspect="equal")

    def plot_pca(self, tfidf=False):

        # Data

        if tfidf:
            matrix = self.__tfidf_matrix
        else:
            matrix = self.__count_matrix

        pca = PCA(n_components=2)
        decomposed_matrix = pca.fit_transform(matrix.A)

        document_word_sum = matrix.sum(axis=1).A.flatten()

        # Setup

        if tfidf:
            document_word_sum_kind = "Document tf-idf sum"
            document_word_sum_format = ".2g"
        else:
            document_word_sum_kind = "Document word count"
            document_word_sum_format = "d"

        # Plotting

        figure = pyplot.figure(
            tight_layout=True,
            figsize=(10, 5),
            # dpi=150
        )
        axis = figure.add_subplot(1, 1, 1)
        seaborn.despine()

        scatter_plot = axis.scatter(
            decomposed_matrix[:, 0], decomposed_matrix[:, 1],
            c=document_word_sum, cmap=COLOUR_MAP,
            zorder=-1
        )

        axis.set_xlabel(
            "Principal component #1 ({:.3g} % of variance explained)"
            .format(pca.explained_variance_ratio_[0] * 100)
        )
        axis.set_ylabel(
            "Principal component #2 ({:.3g} % of variance explained)"
            .format(pca.explained_variance_ratio_[1] * 100)
        )

        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label(document_word_sum_kind)
        colour_bar.ax.zorder = -1

        # Hover annotation

        hover_annotation = axis.annotate(
            s="",
            xy=(0, 0),
            xytext=(0, 20),
            textcoords="offset points",
            bbox={
                "boxstyle": "square",
                "facecolor": "white"
            }
        )
        hover_annotation.set_visible(False)

        def update_hover_annotation(points):

            point_indices = points["ind"]
            number_of_points = len(point_indices)

            positions = numpy.empty((number_of_points, 2))
            texts = []

            for i, index in enumerate(point_indices):

                position = scatter_plot.get_offsets()[index]
                positions[i] = position

                document_number = index + 1
                document = self.__corpus[document_number - 1]
                value = document_word_sum[document_number - 1]

                text = "\n".join([
                    f"Document number: {document_number}.",
                    f"Document content: \"{document}\".",
                    f"{document_word_sum_kind}:"
                    f" {value:{document_word_sum_format}}."
                ])
                texts.append(text)

            hover_annotation.xy = positions.mean(axis=0)
            hover_annotation.set_text("\n\n".join(texts))

        def hover(event):
            visible = hover_annotation.get_visible()
            if event.inaxes == axis:
                contained, points = scatter_plot.contains(event)
                if contained:
                    update_hover_annotation(points)
                    hover_annotation.set_visible(True)
                    figure.canvas.draw_idle()
                elif visible:
                    hover_annotation.set_visible(False)
                    figure.canvas.draw_idle()

        figure.canvas.mpl_connect("motion_notify_event", hover)


def ensure_list_input(value):
    if value and not isinstance(value, list):
        value = [value]
    return value


def ensure_vocabulary_input(value):
    if not isinstance(value, (dict, set)):
        value = ensure_list_input(value)
    return value
