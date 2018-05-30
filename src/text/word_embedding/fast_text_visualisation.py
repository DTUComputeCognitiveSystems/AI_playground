from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.text.word_embedding.fast_text_usage import get_fasttext_model


class WordGroup:
    def __init__(self, name, word_groups, attributes):
        """
        Simple container for holding groups of words for illustrating linear relationships between word embeddings.
        :param str name: Name of this group.
        :param list[list[str]] word_groups: A matrix-like structure of words.
        :param str attributes: Special attributes passed on to visualisation.
        """
        self.name = name
        self.word_groups = word_groups
        self.attributes = attributes

    def __str__(self):
        return "{}({})".format(
            type(self).__name__, self.name
        )

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for item in self.word_groups:
            yield item


def get_test_word_groups():
    """
    Fetch some test-words for illustration found in repository.
    :return:
    """
    _words_file = Path("src", "text", "word_embedding", "word_comparison_lists.txt")

    # Open up
    word_groups = []
    with _words_file.open() as file:
        group = []
        for line in file:

            # Check for blank line (new group)
            if not line.strip():
                if group:
                    # Create word-group container
                    info = group[0]
                    word_group = WordGroup(
                        name=info[0],
                        word_groups=group[1:],
                        attributes="" if len(info) == 1 else info[1]
                    )

                    # Append
                    word_groups.append(word_group)

                # Next group
                group = []

            # Append line to group
            else:
                group.append(line.strip().split(" "))

    # Create and append last group
    if group:
        info = group[0]
        word_group = WordGroup(
            name=info[0],
            word_groups=group[1:],
            attributes="" if len(info) == 1 else info[1]
        )
        word_groups.append(word_group)

    return word_groups


def make_projection_matrix(plane_vectors, method):
    """
    Makes a projection matrix from an iterable of vectors by various methods.
    :param plane_vectors: The vectors. Can be a matrix, in which case the rows are the vectors.
    :param str method: Method used for computing the vectors.
        pca: Use PCA.
        svd: Use SVD on the differences between vectors.
    :return:
    """
    # SVD
    if "svd" in method.lower():
        # Compute differences between pairs of vectors
        vector_differences = np.array([
            group[idx + 1] - group[idx]
            for group in plane_vectors
            for idx in range(len(group) - 1)
        ])

        # Comute mean and SVD
        mean_vector = vector_differences.mean(0)
        svd = np.linalg.svd(vector_differences - mean_vector)

        # Get projection matrix and project vectors
        projection_matrix = svd[2][-2:]

    # PCA
    elif "pca" in method.lower():
        # Flatten vector lists
        vectors_np = np.array(
            [v for item in plane_vectors for v in item]
        )

        # Fit PCA
        pca = PCA(n_components=2)
        _ = pca.fit(vectors_np)

        projection_matrix = pca.components_

    # Unknown method
    else:
        raise ValueError("Unknown methods for plotting embeddings.")

    return projection_matrix


def visualise_vector_pairs(vectors, word_group=None, figsize=(8, 6)):
    """
    Visualize pairs of vectors in a figure. The pairs are generalized to sized more that 2.
    :param vectors: The vectors. Should be a list of iterables, where each iterable holds vectors.
    :param word_group: Words for each vector. If given the words will be visualised instead of scatter points.
    :param figsize:
    :return:
    """
    # Make figure
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # For not plotting text
    if word_group is None:
        word_group = [None] * len(vectors)

    # Plot embeddings
    for pair, words in zip(vectors, word_group):
        # Numpy format
        pair = np.array(pair)

        # Plot texts
        if words is not None:
            for loc, word in zip(pair, words):
                ax.text(
                    x=loc[0],
                    y=loc[1],
                    s=word,
                    ha="center",
                    va="center"
                )
        else:
            ax.scatter(
                pair[:, 0],
                pair[:, 1],
            )

        # Plot lines or points
        if "-" in word_group.attributes:
            ax.plot(pair[:, 0], pair[:, 1])
        else:
            ax.scatter(pair[:, 0], pair[:, 1])

    return fig


def fasttext_projections(word_group, fasttext_model, word_group_for_plane=None, method="pca"):
    """
    Computes a visualisable projection of a group of words' fastText embeddings.
    :param list[list[str]] word_group: Matrix-like structure of words.
    :param fasttext_model:
    :param list[list[str]] word_group_for_plane: Matrix-like structure of words used for computing the plane.
                                 Defaults to the same words as word_group
    :param str method: String passed on to make_projection_matrix.
    :return:
    """
    word_group_for_plane = word_group_for_plane if word_group_for_plane is not None else word_group

    # Make vectors
    plane_vectors = np.array([
        [fasttext_model.get_word_vector(word) for word in group]
        for group in word_group_for_plane
    ])
    if word_group != word_group_for_plane:
        vectors = np.array([
            [fasttext_model.get_word_vector(word) for word in group]
            for group in word_group
        ])
    else:
        vectors = plane_vectors

    # Get projection matrix
    if isinstance(method, str):
        projection_matrix = make_projection_matrix(plane_vectors=plane_vectors, method=method)
    else:
        projection_matrix = method

    # Get projected vectors
    projected_vectors = [[projection_matrix.dot(vec) for vec in item]
                         for item in vectors]

    return projected_vectors


def visualize_word_all_embeddings(word_groups=None, lang="en", method="svd"):
    """
    Visualize embeddings of all words in test-set. Used to test the above methods.
    :param word_groups:
    :param lang:
    :param method:
    :return:
    """
    word_groups = get_test_word_groups() if word_groups is None else word_groups

    # Get fastText model
    model = get_fasttext_model(lang=lang)

    # Go through word groups
    for word_group in word_groups:
        projected_vectors = fasttext_projections(
            word_group=word_group,
            fasttext_model=model,
            method=method
        )
        visualise_vector_pairs(
            vectors=projected_vectors,
            word_group=word_group,
        )


if __name__ == "__main__":
    visualize_word_all_embeddings()
