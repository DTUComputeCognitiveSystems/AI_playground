import feedparser
import pandas as pd
from afinn import Afinn
from ipywidgets.widgets import Accordion, Layout, Label, VBox, HTML, Dropdown, Button, Output
from IPython.core.display import display
from src.text.document_retrieval.wikipedia import Wikipedia
from notebooks.exercises.src.text.news_wiki_search_init import RsspediaInit
from notebooks.exercises.src.text.news_wiki_search import RsspediaSearch

class RSSWikiDashboard:
    def __init__(self, wikipedia:Wikipedia, rsspediainit:RsspediaInit):
        """
        :param Wikipedia wikipedia: wikipedia class object initialized with correct language
        :param RsspediaInit rsspediainit: rsspedia init object - prepares the data and embeddings
        """
        
        self.data_titles = []
        self.data_results = []
        self.rsspediainit = rsspediainit
        self.wikipedia = wikipedia

        self.select_feed = Dropdown(
            options={'Politiken.dk': 0, 
                    'DR.dk': 1, 
                    'BT.dk': 2,
                    'Information.dk': 3,
                    'Børsen.dk': 4,
                    'Ekstrabladet.dk': 5
            },
            value=0,
            description='Vælg nyhedskilde:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.select_search = Dropdown(
            options={'Okapi BM-25': 0, 
                    'Explicit Semantic Analysis': 1,
                    'FTN-a': 2,
                    'FTN-b': 3
            },
            value=0,
            description='Vælg søgnings-algorytme:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.select_nmatches = Dropdown(
            options={'3': 3, 
                    '5': 5,
                    '10': 10
            },
            value=3,
            description='Vælg antal matches:',
            disabled=False,
            layout=Layout(width='400px'),
            style={'description_width': '160px'},
        )

        self.container = Output(
            value = "",
        )

        self.submit_button = Button(
            value=False,
            description='Indlæs nyheder',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Få nyheder fra RSS-feed og find Wikipedia matches',
            icon=''
        )

        self.widget_box = VBox(
            (self.select_feed, 
             self.select_search, 
             self.select_nmatches,
             self.submit_button, 
             self.container),
        )

        self.submit_button.on_click(self._run)

    @property
    def start(self):
        return self.widget_box

    def display_beautifully(self):
        content_items = []
        for i in range(len(self.data_results)):
            news, titles, texts, urls, scores = self.data_results[i]
            #content_items.append(self.rsspedia_search.display_beautifully(titles, texts, urls, scores))
            content_items.append(HTML(value = "{}".format(self.rsspedia_search.display_beautifully(titles, texts, urls, scores))))
        accordion = Accordion(children = (content_items),)

        for i in range(len(self.data_titles)):
            accordion.set_title(i, "{}. {}".format(i + 1, self.data_titles[i]))

        display(accordion)

    def _run(self, *args, **kwargs):
        selected_value = False
        self.data_results = []
        self.data_titles = []
        RSS_feeds = [('Politiken.dk', 'http://politiken.dk/rss/senestenyt.rss'), 
                    ('DR.dk', 'http://www.dr.dk/Forms/Published/rssNewsFeed.aspx?config=6b82610d-b898-49b2-80ef-85c5642519c3&rss=Yes&rssTitle=DR+Nyheder+Online&overskrift=Politik+-+seneste+20&Url=%2fnyheder%2f'), 
                    ('BT.dk', 'https://www.bt.dk/bt/seneste/rss'),
                    ('Information.dk', 'https://www.information.dk/feed'),
                    ('Børsen.dk', 'https://borsen.dk/rss/'),
                    ('Ekstrabladet.dk', 'http://ekstrabladet.dk/seneste10.rss')
        ]

        search_algorithms = [("okapibm25"),
                             ("esa_relatedness"),
                             ("fasttext_a"),
                             ("fasttext_b")
        ]
        search_algorithm = search_algorithms[self.select_search.value]

        self.rsspedia_search = RsspediaSearch(rsspediainit = self.rsspediainit, wikipedia = self.wikipedia)
        

        if selected_value:
            feed = feedparser.parse(RSS_feeds[selected_value][1])
        else:
            feed = feedparser.parse(RSS_feeds[self.select_feed.value][1])


        # Get relevant objects from RSS feed ans store titles and scores
        for i in range(len(feed["entries"])):
            self.data_titles.append(feed["entries"][i]["title"])
            titles, texts, urls, scores = self.rsspedia_search.search_wiki(search_texts = [feed["entries"][i]["title"]], n_matches = self.select_nmatches.value, search_type = search_algorithm)
            self.data_results.append([feed["entries"][i]["title"], titles, texts, urls, scores])
        
        return self.display_beautifully()