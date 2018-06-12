import numpy as np

from src.text.document_retrieval.wikipedia import Wikipedia

class WikipediaSearcher:

    def __init__(self, wikipedia: Wikipedia):
        self.wikipedia = wikipedia

    def __str__(self):
        return "{}({} documents, {} words)".format(type(self).__name__, self.n_documents, self.n_words)

    def __repr__(self):
        return str(self)

    def search(self, search, k=1.5, b=0.75, search_min_threshold=0, max_results=None):

        if isinstance(search, list):
            search = "\n\n".join(search)

        # Vectorize search
        m_search_term = self.wikipedia.term_vectorizer.transform([search])
        m_search_bool = (m_search_term > 0).toarray()[0, :]

        # Get term-frequencies of query for all documents
        doc_f = self.wikipedia.matrix_doc_term[:, m_search_bool]

        # Get IDF of the search words
        idf_search = self.wikipedia.idf[m_search_bool]

        # Compute okapi bm25 score
        numerator_matrix = doc_f * (k + 1)
        denominator_matrix = doc_f + k * (1 - b + b *
            (self.wikipedia.document_lengths
            / self.wikipedia._avg_document_length))
        fraction_matrix = numerator_matrix / denominator_matrix
        scores = np.squeeze(
            np.array(
                fraction_matrix @ np.expand_dims(idf_search, axis=1)
            ),
            axis=1
        )

        # Sort search results
        sort_locs = np.argsort(a=-scores, kind="quicksort")
        sort_scores = scores[sort_locs]

        # Remove zero-scores
        thresh_locs = sort_scores > search_min_threshold
        sort_locs = sort_locs[thresh_locs]
        sort_scores = sort_scores[thresh_locs]

        # Weave indices and scores for output
        output = [(idx, c_score) for idx, c_score in zip(sort_locs, sort_scores)]

        # Restrict number of outputs
        if isinstance(max_results, int):
            output = output[:max_results]

        return output


if __name__ == "__main__":
    wikipedia = Wikipedia()
    searcher = WikipediaSearcher(wikipedia)
