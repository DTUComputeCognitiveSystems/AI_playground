from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.text.word_embedding.fast_text_usage import get_fasttext_model


class WordGroup:
    def __init__(self, name, word_groups, attributes):
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


def visualize_word_embeddings(word_groups=None, lang="en", method="pca"):
    word_groups = get_test_word_groups() if word_groups is None else word_groups

    # Get fastText model
    model = get_fasttext_model(lang=lang)

    # Go through word groups
    for word_group in word_groups:
        # Make figure
        _ = plt.figure()
        ax = plt.gca()

        # Make vectors
        vectors = np.array([
            [model.get_word_vector(word) for word in group]
            for group in word_group
        ])

        # SVD
        if "svd" in method.lower():
            # Compute differences between pairs of vectors
            vector_differences = np.array([
                group[idx + 1] - group[idx]
                for group in vectors
                for idx in range(len(group) - 1)
            ])

            # Comute mean and SVD
            mean_vector = vector_differences.mean(0)
            svd = np.linalg.svd(vector_differences - mean_vector)

            # Get projection matrix and project vectors
            projection_matrix = svd[2][-2:]
            pro_vectors = [[projection_matrix.dot(vec) for vec in item]
                           for item in vectors
                           ]

        # PCA
        elif "pca" in method.lower():
            # Flatten vector lists
            vectors_np = np.array(
                [v for item in vectors for v in item]
            )

            # Fit PCA
            pca = PCA(n_components=2)
            _ = pca.fit(vectors_np)

            pro_vectors = [[pca.transform(np.expand_dims(vec, 0))[0, :] for vec in item]
                           for item in vectors
                           ]

        # Unknown method
        else:
            raise ValueError("Unknown methods for plotting embeddings.")

        # Plot embeddings
        for pair, words in zip(pro_vectors, word_group):
            # Numpy format
            pair = np.array(pair)

            # Plot texts
            for loc, word in zip(pair, words):
                ax.text(
                    x=loc[0],
                    y=loc[1],
                    s=word,
                    ha="center",
                    va="center"
                )

            # Plot lines or points
            if "-" in word_group.attributes:
                ax.plot(pair[:, 0], pair[:, 1])
            else:
                ax.scatter(pair[:, 0], pair[:, 1])


if __name__ == "__main__":
    visualize_word_embeddings()
