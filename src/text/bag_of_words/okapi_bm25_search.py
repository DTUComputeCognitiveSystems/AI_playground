import numpy
import pandas
import scipy.sparse

class OkapiBM25Searcher:
    def __init__(self,
                 tf_matrix: scipy.sparse.csr_matrix,
                 idf_vector: numpy.array):

        self.tf_matrix = tf_matrix
        self.idf_vector = idf_vector

        document_lengths = self.tf_matrix.sum(axis=1)
        average_document_length = document_lengths.mean()
        self.normalised_document_lengths = \
            document_lengths / average_document_length

    def search(self, query_vectorised, k_1=1.2, b=0.75):
        
        # Find keyword indices
        _, keyword_indices = query_vectorised.nonzero()

        # Term frequencies of keywords for all documents
        tf_matrix = self.tf_matrix[:, keyword_indices]

        # IDF of keywords
        idf_vector = numpy.expand_dims(
            self.idf_vector[keyword_indices],
            axis=1
        )

        # Normalised document lengths
        l = self.normalised_document_lengths

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

        # Compute fraction in Okapi score equation
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
        results = pandas.Series(scores.flatten())
        
        # Remove zero-scores
        results = results[results > 0]
        
        # Sort results
        results = results.sort_values(
            ascending=False,
            kind="quicksort"
        )

        return results
