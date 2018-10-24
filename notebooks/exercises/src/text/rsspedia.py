""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())
    
import re
from src.text.document_retrieval.wikipedia import Wikipedia

class Rsspedia:
    def __init__(self, wikipedia: Wikipedia):

        self.wikipedia = wikipedia

        self.search_results = []
        self.content = self.texts = self.wikipedia_results = None

    def search_wikipedia(self, search_texts, k_1=1.2, b=0.75):

        if search_texts:
            for i, text in enumerate(search_texts):
                wikipedia_results, search_terms = self.wikipedia.search(
                    query=text,
                    k_1=k_1,
                    b=b
                )
                formatted_result_list = ["<ol>"]
                for index, score in wikipedia_results[:3].items():
                    document = self.wikipedia.documents[index]
                    title = document.title
                    WIKIPEDIA_DOCUMENT_TITLE_PREFIX = "Wikipedia: "
                    if title.startswith(WIKIPEDIA_DOCUMENT_TITLE_PREFIX):
                        title = title.replace(
                            WIKIPEDIA_DOCUMENT_TITLE_PREFIX,
                            "",
                            1
                        )
                    abstract = document.abstract
                    for search_term in search_terms:
                        abstract = re.sub(
                            pattern=r"(\b" + search_term + r"\b)",
                            repl=r"<b>\1</b>",
                            string=abstract,
                            flags=re.IGNORECASE
                        )
                    url = document.url
                    formatted_result = "\n".join([
                        "<li>",
                        f"<p><a href=\"{url}\">{title}</a></p>",
                        f"<p>{abstract}</p>",
                        f"<p><b>Score: {score:.3g}</b></p>",
                        "</li>"
                    ])
                    formatted_result_list.append(formatted_result)
                formatted_result_list.append("</ol>")
                formatted_results = "\n".join(formatted_result_list)
                self.search_results.append(formatted_results)

    def loadTexts(self, texts):
        self.texts = texts

    def display_beautifully(self, titles, texts, urls):
        formatted_result_list = ["<ol>"]
        for i in range(len(titles)):
            formatted_result = "\n".join([
                "<li>",
                f"<p><a href=\"{urls[i]}\">{titles[i]}</a></p>",
                f"<p>{texts[i]}</p>",
                "</li>"
            ])
            formatted_result_list.append(formatted_result)
        formatted_result_list.append("</ol>")
        formatted_results = "\n".join(formatted_result_list)
        return formatted_results

    def search_wiki(self, search_texts, n_matches = 3, search_type = "okapibm25"):
        k_1 = 1.2
        b = 0.75

        titles = texts = urls = []

        if search_texts:
            for i, text in enumerate(search_texts):
                wikipedia_results, search_terms = self.wikipedia.search(
                    query=text,
                    k_1=k_1,
                    b=b
                )

                for index, score in wikipedia_results[:3].items():
                    document = self.wikipedia.documents[index]

                    titles.append(document.title)
                    texts.append(document.abstract)
                    urls.append(document.url)

        return titles, texts, urls

if __name__ == "__main__":
    print(1)