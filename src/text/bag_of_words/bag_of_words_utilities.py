import numpy
import pandas
import scipy.sparse
import seaborn
from matplotlib import pyplot
from wordcloud import WordCloud
from sklearn.decomposition import PCA


COLOUR_MAP = seaborn.cubehelix_palette(as_cmap=True)
MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES = 25
MAXIMUM_NUMBER_OF_DOCUMENTS_FOR_SIMPLE_PLOT = 10
MAXIMUM_NUMBER_OF_WORDS_FOR_SIMPLE_PLOT = 25
MAXIMUM_VALUE_RANGE_FOR_DISCRETE_COLOUR_BAR = 25

ALLOWED_KINDS_OF_BAG_OF_WORDS = ["count", "tfidf"]


def bag_of_words_as_data_frame(matrix, terms, document_ids=None,
                            remove_absent_terms=False):

    if terms is None:
        raise ValueError(
            "No terms were provided for bag-of-words matrix."
        )

    number_of_documents = matrix.shape[0]

    terms = numpy.array(terms)

    if remove_absent_terms:
        term_sums = matrix.sum(axis=0).A.flatten()
        absent_term_indices = term_sums != 0
        matrix = matrix[:, absent_term_indices]
        terms = terms[absent_term_indices]

    terms = terms.tolist()

    if document_ids is None:
        document_ids = [i + 1 for i in range(number_of_documents)]

    data_frame = pandas.SparseDataFrame(
        matrix,
        index=document_ids,
        columns=terms,
        default_fill_value=0
    )
    data_frame = data_frame.astype(matrix.dtype)

    return data_frame


def kind_of_bag_of_words(bag_of_words):

    minimum = bag_of_words.min()
    maximum = bag_of_words.max()
    dtype = bag_of_words.dtype

    kind = None

    if minimum >=0:
        if (bag_of_words == bag_of_words.astype(int)).all():
            kind = "count"
        elif "float" in dtype.name and maximum <= 1:
            kind = "tfidf"

    if not kind:
        raise ValueError("Input is not a bag-of-words matrix.")

    return kind


def ensure_kind_of_bag_of_words(kind):

    if kind:

        kind = kind.replace("-", "").lower()

        if kind not in ALLOWED_KINDS_OF_BAG_OF_WORDS:
            allowed_kinds = ", ".join(map(lambda s: f"\"{s}\"",
                ALLOWED_KINDS_OF_BAG_OF_WORDS))
            raise ValueError(
                f"`kind` can only be {allowed_kinds}, or `None`"
            )

    return kind


def plot_heat_map_of_bag_of_words(bag_of_words, kind=None, terms=None,
                                  document_ids=None, corpus=None,
                                  remove_absent_terms=None):
    
    kind = ensure_kind_of_bag_of_words(kind)

    if kind is None:
        kind = kind_of_bag_of_words(bag_of_words)

    # Data

    number_of_documents, number_of_terms = bag_of_words.shape

    if not isinstance(bag_of_words, pandas.DataFrame):

        if remove_absent_terms is None:
            if number_of_terms \
                    > MAXIMUM_NUMBER_OF_WORDS_WHEN_INCLUDING_ABSENT_ONES:
                remove_absent_terms = True
            else:
                remove_absent_terms = False

        bag_of_words = bag_of_words_as_data_frame(
            matrix=bag_of_words,
            terms=terms,
            document_ids=document_ids,
            remove_absent_terms=remove_absent_terms
        )
        bag_of_words = bag_of_words.reindex(index=bag_of_words.index[::-1])
        bag_of_words = bag_of_words.to_dense()

    # Setup

    if number_of_documents \
            <= MAXIMUM_NUMBER_OF_DOCUMENTS_FOR_SIMPLE_PLOT \
            and number_of_terms \
            <= MAXIMUM_NUMBER_OF_WORDS_FOR_SIMPLE_PLOT:
        annotate = True
        line_width = 1
        y_tick_labels = "auto"
        colour_bar = False
        colour_bar_parameters = {}
    else:
        annotate = False
        line_width = 0
        y_tick_labels = max(number_of_documents // 4, 5)
        colour_bar = True
        colour_bar_parameters = {
            "orientation": "horizontal"
        }

    colour_bar_tick_labels = None

    if kind == "tfidf":
        value_kind = "tf-idf value"
        annotation_format = ".2g"
    elif kind == "count":
        value_kind = "Count"
        annotation_format = "d"

        minimum_value = bag_of_words.values.min()
        maximum_value = bag_of_words.values.max()
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
        bag_of_words,
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

    axis.set_xlabel("Terms")
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
        document_number = number_of_documents - document_index
        document_content = None
        term_index = int(x)
        term = bag_of_words.columns[term_index]
        value = bag_of_words.values[document_index, term_index]

        if corpus:
            document_content = corpus[document_number - 1]

        text = "\n".join([
            f"Document number: {document_number}.",
            f"Document content: \"{document_content}\"." if document_content else "",
            f"Term: \"{term}\".",
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


def plot_term_cloud_for_bag_of_words(bag_of_words_matrix, idf=None,
                                     terms=None):

    if idf is None:
        sum_vector = scipy.sparse.coo_matrix(bag_of_words_matrix.sum(axis=0))
        frequency_vector = sum_vector / sum_vector.max()

    else:
        nonzero_indices = bag_of_words_matrix.sum(axis=0).nonzero()
        idf_normalised = (idf - idf.min()) / (idf.max() - idf.min())
        frequency_vector = scipy.sparse.coo_matrix((
            idf_normalised[nonzero_indices[1]], nonzero_indices
        ))

    term_frequencies = {}

    for index, value in zip(frequency_vector.col,
                            frequency_vector.data):
        term = terms[index]
        frequency = value
        term_frequencies[term] = frequency

    term_cloud_generator = WordCloud(
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
    term_cloud = term_cloud_generator.generate_from_frequencies(
        term_frequencies)

    figure = pyplot.figure(
        figsize=(10, 5),
        # dpi=150
        frameon=False
    )
    axis = pyplot.Axes(figure, [0., 0., 1., 1.])
    axis.set_axis_off()
    figure.add_axes(axis)

    axis.imshow(term_cloud, interpolation="bilinear", aspect="equal")


def plot_pca_of_bag_of_words(bag_of_words_matrix, kind=None,
                             document_ids=None, corpus=None):

    kind = ensure_kind_of_bag_of_words(kind)

    if kind is None:
        kind = kind_of_bag_of_words(bag_of_words_matrix)

    # Data

    pca = PCA(n_components=2)
    decomposed_bag_of_words_matrix = pca.fit_transform(bag_of_words_matrix.A)

    document_term_sum = bag_of_words_matrix.sum(axis=1).A.flatten()

    # Setup

    if kind == "tfidf":
        document_term_sum_kind = "Document tf-idf sum"
        document_term_sum_format = ".2g"
    elif kind == "count":
        document_term_sum_kind = "Document word count"
        document_term_sum_format = "d"

    # Plotting

    figure = pyplot.figure(
        tight_layout=True,
        figsize=(10, 5),
        # dpi=150
    )
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    scatter_plot = axis.scatter(
        decomposed_bag_of_words_matrix[:, 0],
        decomposed_bag_of_words_matrix[:, 1],
        c=document_term_sum, cmap=COLOUR_MAP,
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
    colour_bar.set_label(document_term_sum_kind)
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
            document_content = None

            if corpus:
                document_content = corpus[document_number - 1]

            value = document_term_sum[document_number - 1]

            text = "\n".join([
                f"Document number: {document_number}.",
                f"Document content: \"{document_content}\".",
                f"{document_term_sum_kind}:"
                f" {value:{document_term_sum_format}}."
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
