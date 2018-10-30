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
        self.data_titles = []
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

    def _run(self, selected_value = None):
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

        rsspedia_search = RsspediaSearch(rsspediainit = self.rsspediainit, wikipedia = self.wikipedia)

        if selected_value:
            feed = feedparser.parse(RSS_feeds[selected_value][1])
        else:
            feed = feedparser.parse(RSS_feeds[self.select_feed.value][1])


        # Get relevant objects from RSS feed ans store titles and scores
        for i in range(len(feed["entries"])):
            self.data_titles.append(feed["entries"][i]["title"])


