import numpy as np
from gensim.models.fasttext import FastText
from scipy.spatial.distance import cdist
import re
from src.text.document_retrieval.wikipedia import Wikipedia
from notebooks.exercises.src.text.news_wiki_search_init import RsspediaInit


class RsspediaSearch:
    def __init__(self, wikipedia: Wikipedia, rsspediainit: RsspediaInit):
        """
        :param Wikipedia wikipedia: wikipedia class object initialized with correct language
        :param RsspediaInit rsspediainit: rsspedia init object - prepares the data and embeddings
        """

        self.rsspediainit = rsspediainit
        self.embedding_composition = rsspediainit.embedding_composition
        self.search_results = []
        self.content = self.wikipedia_results = None
        # Initialize wikipedia
        self.wikipedia = wikipedia
        
        #self.texts = rsspediainit.texts
        
        # Calculate tf-idf representation for wiki texts (takes time)
        self._transformer = rsspediainit._transformer
        self._Y = rsspediainit._Y
        
        # Fasttext: initialize the model
        self.model = rsspediainit.model
        
        #self.wikipedia.documents_clean 
        self.wikipedia_abstract_vectors = rsspediainit.wikipedia_abstract_vectors
        self.wikipedia_title_vectors = rsspediainit.wikipedia_title_vectors
    
    def get_ngrams(self, text):
        words = self.rsspediainit.getCleanWordsList(text)
        words_copy = words.copy()

        for i in range(len(words_copy)):
            if i > 0:
                words_copy.append(words_copy[i - 1] + " " + words_copy[i])

        return words_copy
    
    def cdist_func(self, A, B):
        # Calculates cosine distance
        dists = cdist(A, B, 'cosine')
        return np.argmin(dists, axis=0), dists #np.min(dists, axis=0)

    def display_beautifully(self, titles, texts, urls, scores):
        formatted_result_list = ["<ol>"]
        for i in range(len(titles)):
            formatted_result = "\n".join([
                "<li>",
                f"<p><a href=\"{urls[i]}\">{round(scores[i], 3)} {titles[i]}</a></p>",
                f"<p>{texts[i]}</p>",
                "</li>"
            ])
            formatted_result_list.append(formatted_result)
        formatted_result_list.append("</ol>")
        formatted_results = "\n".join(formatted_result_list)
        return formatted_results

    def search_wiki(self, search_texts, n_matches = 3, search_type = "okapibm25", remove_similar = False, verbose = False, p = 0.5):
        n_mult_factor = 3 # factor to multiply n_matches with
        n_matches = n_matches * 3 # this is done to remove very similar values from the results and ensure we have enough to return
        titles = [] 
        texts = []
        urls = []
        scores = []
        
        # (1) Remove unnecessary symbols from the search text
        pattern = re.compile('[^a-zA-Z0-9åÅøØæÆ ]+', re.UNICODE)

        if search_texts:
            for i, text in enumerate(search_texts):
                # (1) Remove unnecessary symbols from the search text
                text = pattern.sub('', text)
                
                if search_type == "okapibm25":
                    wikipedia_results, search_terms = self.wikipedia.search(query = text, k_1 = 1.2, b = 0.75)
                    for index, score in wikipedia_results[:n_matches].items():
                        document = self.wikipedia.documents[index]
                        titles.append(document.title)
                        texts.append(document.abstract)
                        urls.append(document.url)
                        scores.append(score)
                elif search_type == "esa_relatedness":
                    y = self._transformer.transform([text])
                    D = np.array((self._Y * y.T).todense())
                    indices = np.argsort(-D, axis=0)
                    titles = [self.wikipedia.documents[index].title for index in indices[:n_matches, 0]]
                    texts = [self.wikipedia.documents[index].abstract for index in indices[:n_matches, 0]]
                    urls = [self.wikipedia.documents[index].url for index in indices[:n_matches, 0]]
                    scores = [0.0 for i in range(len(texts))]
                elif search_type == "fasttext_a":
                    # Calculate the vectorized representation
                    text_vector = self.rsspediainit.sumVectorRepresentation(text = text, type = self.embedding_composition)
                    # Calculate the distance between the wiki abstract vectors and the searchable texts
                    cdist_result = self.cdist_func(self.wikipedia_abstract_vectors, [text_vector])
                    cdist_list = cdist_result[1] # List of all the cosine distances
                    cdist_list_sorted = np.sort(cdist_list, axis = 0) # Sorted list of cosine distances - to get top N matches

                    for i in range(n_matches):
                        result = np.where(cdist_list == cdist_list_sorted[i])
                        document = self.wikipedia.documents_clean[result[0][0]]
                        titles.append(document.title)
                        texts.append(document.abstract)
                        urls.append(document.url)
                        scores.append(cdist_list[result][0])
                elif search_type == "fasttext_b":
                    ngrams = self.get_ngrams(text)
                    r = []
                    for i in range(len(ngrams)):
                        # First, compute the distance between the current n-gram and the title of the wikipedia article.
                        cdist_result = self.cdist_func(self.wikipedia_title_vectors, [self.rsspediainit.sumVectorRepresentation(text = ngrams[i], type = self.embedding_composition)])
                        # Second, compute the distance between the current n-gram and the search text.
                        cdist_result2 = self.cdist_func([self.rsspediainit.sumVectorRepresentation(text = text, type = self.embedding_composition)], [self.rsspediainit.sumVectorRepresentation(text = ngrams[i], type = self.embedding_composition)])

                        cdist_list1 = cdist_result[1] # List of all the cosine distances
                        cdist_list2 = cdist_result2[1]
                        # Third, combine the two cosine distances using the p parameter that takes values in this range: (0,1)
                        cdist_list = (cdist_list1 * p + cdist_list2 * (1 - p))
                        cdist_list_sorted = np.sort(cdist_list, axis = 0) # Sorted list of cosine distances - to get top N matches
                        
                        for j in range(n_matches):
                            x = np.where(cdist_list == cdist_list_sorted[j])[0]
                            # Check if the result is empty, then we just skip it. 
                            # This means that one of the n-grams yielded a vector embedding filled with zeros.
                            # Thus, the cosine distance is undefined.
                            if(cdist_list[x].any()):
                                r.append( (x, cdist_list[x][0]))
                            else:
                                continue
   
                            if verbose:
                                print("{} {} {} {}".format(x, self.wikipedia.documents_clean[x[0]].title, cdist_list[x], ngrams[i]))

                    # When np.where returns multiple matches, we flatten them
                    # Example: r[13] = [123, 1413], and we remove r[13] and add r[n+1] = 123, r[n+2] = 1413
                    r_copy = r.copy()
                    uniques = []
                    for i in range(len(r)-1, -1, -1):
                        if len(r[i][0]) > 1:
                            r_copy.pop(i)
                            for j in range(len(r[i][0])):
                                r_copy.append( (np.array([r[i][0][j]]), r[i][1]))

                    # Remove duplicate wikipedia pages. They occur because different n-grams can match the same pages
                    for i in range(len(r_copy)-1,-1,-1):
                        if r_copy[i][0] in uniques:
                            r_copy.pop(i)
                        else:
                            uniques.append(r_copy[i][0])
                    
                    r = r_copy
                    # Transform into list of tuples
                    r = [ (r[i][0][0], r[i][1][0]) for i in range(len(r))]
                    # Sort the list of tuples by cosine distance
                    r = sorted(r, key=lambda tup: tup[1])
                    
                    for i in range(len(r)):
                        document = self.wikipedia.documents_clean[r[i][0]]
                        titles.append(document.title)
                        #print("{} {}".format(document.title, r[i][1]))
                        texts.append(document.abstract)
                        urls.append(document.url)
                        scores.append(r[i][1])
            
            if remove_similar:
                # Removing too similar search results
                # Get vectors of the result titles
                title_result_vectors = [self.rsspediainit.sumVectorRepresentation(text = titles[i], type = self.embedding_composition) for i in range(len(titles))]
                titles_pruned = titles.copy()
                n_removed = 0
                ids_removed = []
                for i in range(len(titles)):
                    # Get cosine distances
                    cdist_result = self.cdist_func(title_result_vectors, [self.rsspediainit.sumVectorRepresentation(text = titles[i], type = self.embedding_composition)])[1]
                    # Sort cosine distances
                    cdist_result_sorted = np.sort(cdist_result, axis = 0)
                    rd = []
                    for j in range(len(titles) - i):
                        if i != j + i:
                            x = np.where(cdist_result == cdist_result_sorted[j + i])[0]
                            rd.append( (x, cdist_result[x][0]))
                            if cdist_result[x][0] < 0.10 and i + j not in ids_removed:
                                titles_pruned.pop(i + j - n_removed)
                                texts.pop(i + j - n_removed)
                                urls.pop(i + j - n_removed)
                                scores.pop(i + j - n_removed)
                                n_removed = n_removed + 1
                                ids_removed.append(i + j)
                                #print("removed: {}".format(i + j))
                            #print("{}-th title: {}, {}-th title: {}, dist: {}".format(i, titles[i], j + i, titles[j + i], cdist_result[x]))
                titles = titles_pruned
        return titles[:int(n_matches / n_mult_factor)], texts[:int(n_matches / n_mult_factor)], urls[:int(n_matches / n_mult_factor)], scores[:int(n_matches / n_mult_factor)]

