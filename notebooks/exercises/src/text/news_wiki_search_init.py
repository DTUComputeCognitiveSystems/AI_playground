from gensim.models.fasttext import FastText
from scipy.spatial.distance import cdist
import re
from pathlib import Path
from src.text.document_retrieval.wikipedia import Wikipedia
import numpy as np
# ESA relatedness package
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utility.files import (
    ensure_directory,
    save_as_compressed_pickle_file,
    load_from_compressed_pickle_file
)

DATA_DIRECTORY = Path("data", "wikipedia")
ensure_directory(DATA_DIRECTORY)


class RsspediaInit:
    def __init__(self, wikipedia: Wikipedia, embedding_composition = "sum"):
        """
        :param Wikipedia wikipedia: wikipedia class object initialized with correct language
        :param str embedding_composition: "sum" or "average" the embeddings to calculate the representation
        """

        self.embedding_composition = embedding_composition
        self.search_results = []
        self.content = self.wikipedia_results = None
        # Initialize wikipedia
        self.wikipedia = wikipedia
        self.wikipedia.documents_clean = self.wikipedia.documents.copy()
        self.wikipedia_abstract_vectors = []
        self.wikipedia_title_vectors = []

        # self.texts = []
        # self.texts_clean = []
        # self._transformer = []
        # self._Y = []
        # self.model = []

        print("\n")

        self.__fasttext_abstracts_titles_documents_filename = "da-fasttext-vectorized-abstracts-titles.pkl.gz"
        self.__vectorised_fasttext_documents_path = Path(
            DATA_DIRECTORY, self.__fasttext_abstracts_titles_documents_filename)

        self.__tfidf_texts_documents_filename = "da-tfidf-vectorized-texts.pkl.gz"
        self.__vectorised_tfidf_documents_path = Path(
            DATA_DIRECTORY, self.__tfidf_texts_documents_filename)

        # Calculate tf-idf representation for wiki texts (takes time)
        
        if self.__vectorised_tfidf_documents_path.exists():
            print("Loading vectorized TF-IDF documents.")
            self.vectorised_storage_tfidf = load_from_compressed_pickle_file(
            self.__vectorised_tfidf_documents_path)
            
            self._transformer = self.vectorised_storage_tfidf["content"][0]
            self._Y = self.vectorised_storage_tfidf["content"][1]

            print("Vectorized TF-IDF documents loaded.")
        else:
            print("Computing vectorized TF-IDF documents.")
            self._transformer = TfidfVectorizer(stop_words = None, norm = "l2", use_idf = True, sublinear_tf = False)

            # Remove all the line breaks and caret returns from wiki texts
            pattern = re.compile('[\n\r ]+', re.UNICODE)
            self.texts = [self.wikipedia.documents[i].text for i in range(len(self.wikipedia.documents))]
            self.texts = [pattern.sub(' ', self.texts[i]) for i in range(len(self.texts))]
            self.texts_clean = [self.getCleanWordsList(self.texts[i], return_string = True) for i in range(len(self.texts))]
        
            self._Y = self._transformer.fit_transform(self.texts_clean)

            parsed_documents = {
                "content": (self._transformer, self._Y),
                "changes": "TF-IDF vectorized representation of the wikipedia article full texts (Danish)",
                "license": "None"
            }
            save_as_compressed_pickle_file(
                parsed_documents,
                self.__vectorised_tfidf_documents_path
            )
            print("Vectorized TF-IDF documents computed and saved into file.")

        print("\n")

        # Fasttext: initialize the model
        print("Initializing and loading Fasttext binaries.")
        bin_path = Path("data", "fasttext", "wiki.da.bin")
        self.model = FastText.load_fasttext_format(str(bin_path))
        print("Fasttext initialized and loaded.")
        
        print("\n")
        
        # # Fasttext: Compute vectorized representation for all wikipedia articles (takes time)
        if self.__vectorised_fasttext_documents_path.exists():
            print("Loading vectorized Fasttext documents.")
            self.vectorised_storage_fasttext = load_from_compressed_pickle_file(
            self.__vectorised_fasttext_documents_path)

            self.wikipedia_abstract_vectors = self.vectorised_storage_fasttext["content"][0]
            self.wikipedia_title_vectors = self.vectorised_storage_fasttext["content"][1]
            self.wikipedia.documents_clean = self.vectorised_storage_fasttext["content"][2]

            print("Vectorized Fasttext documents loaded.")
        else:
            print("Computing vectorized Fasttext documents.")
            i = 0
            i_max = 0
            n_removed = 0
            pattern1 = re.compile('[^a-zA-Z0-9åÅøØæÆ]+', re.UNICODE)

            for n in range(len(self.wikipedia.documents)):
                # if abstract length is zero, remove it
                try:
                    if len(pattern1.sub('', self.wikipedia.documents[n].abstract)) == 0:
                        del self.wikipedia.documents_clean[n - n_removed]
                        n_removed = n_removed + 1
                    else:
                        self.wikipedia_abstract_vectors.append(self.sumVectorRepresentation(text = self.wikipedia.documents[n].abstract, type = self.embedding_composition))
                        self.wikipedia_title_vectors.append(self.sumVectorRepresentation(text = self.wikipedia.documents[n].title, type = self.embedding_composition))
                        
                        i = i + 1
                        if i_max > 0 and i > i_max:
                            break
                except IndexError:
                    print("n: {}, n_removed: {}, w.d: {}, w.d_c: {}".format(n, n_removed, len(self.wikipedia.documents), len(self.wikipedia.documents_clean)))

            parsed_documents = {
                "content": (self.wikipedia_abstract_vectors, self.wikipedia_title_vectors, self.wikipedia.documents_clean),
                "changes": "Abstrats and titles of wikipedia articles (Danish)",
                "license": "None"
            }
            save_as_compressed_pickle_file(
                parsed_documents,
                self.__vectorised_fasttext_documents_path
            )
            print("Vectorized Fasttext documents computed and saved into file.")

    def getCleanWordsList(self, text, return_string = False):
        pattern = re.compile('[^a-zA-Z0-9åÅøØæÆ ]+', re.UNICODE)
        text = pattern.sub('', text)
        words = text.lower().strip().split()
        words_copy = words.copy()
        #stop_words = ["den", "det", "en", "et", "om", "for", "til", "at", "af", "på", "som", "og", 
        #              "jeg", "mig", "mine", "min", "mit", "du", "dig", "din", "dit", "dine", "han", "ham", "hun", "hende", 
        #              "de", "dem", "vi", "os", "sin", "sit", "sine", "sig"]
        
        stop_words = ["den", "det", "denne", "dette", "en", "et", "om", "for", "til", "at", "af", "på", "som", "og", "er", "i"]
        
        n_removed = 0
        for i in range(len(words)):
            if words[i] in stop_words:
                words_copy.pop(i - n_removed)
                n_removed = n_removed + 1
        if return_string:
            return ' '.join(words_copy)
        else:
            return words_copy
    
    def sumVectorRepresentation(self, text, verbose = False, type = "sum"):
        # Calculates vectorized represetnation of some text
        words = self.getCleanWordsList(text)
        text_vector = np.zeros(self.model.wv["a"].shape)
        if verbose:
            print("len: {}, words: {}".format(len(words), words))
        for i in range(len(words)):
            try:
                if type == "average":
                    text_vector = text_vector + self.model.wv[words[i]] / len(words)
                else: #sum
                    text_vector = text_vector + self.model.wv[words[i]]
            except KeyError as e:
                if verbose:
                    print("i: {}, e: {}".format(i, e))
                continue
        return text_vector

