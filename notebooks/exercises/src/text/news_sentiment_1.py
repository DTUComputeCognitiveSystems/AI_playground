import feedparser
import pandas as pd
from afinn import Afinn
from ipywidgets.widgets import Accordion, Layout, Label, VBox, HTML, Dropdown, Button, Output
from IPython.core.display import display
from notebooks.exercises.src.text.rsspedia import Rsspedia

class RSSDashboard:
    def __init__(self):
        self.data_titles = []
        self.data_scores = []
        self.afinne = None

        self.select = Dropdown(
            options={'Politiken.dk': 0, 
                    'DR.dk': 1, 
                    'BT.dk': 2,
                    'Information.dk': 3,
                    'Børsen.dk': 4,
                    'Ekstrabladet.dk': 5
            },
            value=1,
            description='Vælg nyhedskilde:',
            disabled=False,
            layout=Layout(width='300px'),
            style={'description_width': '130px'},
        )

        self.container = Output(
            value = "",
        )

        self.submit_button = Button(
            value=False,
            description='Indlæs nyheder',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Få nyheder fra RSS-feed og lav sentiment-analyse',
            icon=''
        )

        self.widget_box = VBox(
            (self.select, self.submit_button, self.container),
        )

        self.submit_button.on_click(self._do_sentiment_analysis)

    @property
    def start(self):
        return self.widget_box

    def _do_sentiment_analysis(self, _):
        RSS_feeds = [('Politiken.dk', 'http://politiken.dk/rss/senestenyt.rss'), 
                    ('DR.dk', 'http://www.dr.dk/Forms/Published/rssNewsFeed.aspx?config=6b82610d-b898-49b2-80ef-85c5642519c3&rss=Yes&rssTitle=DR+Nyheder+Online&overskrift=Politik+-+seneste+20&Url=%2fnyheder%2f'), 
                    ('BT.dk', 'https://www.bt.dk/bt/seneste/rss'),
                    ('Information.dk', 'https://www.information.dk/feed'),
                    ('Børsen.dk', 'https://borsen.dk/rss/'),
                    ('Ekstrabladet.dk', 'http://ekstrabladet.dk/seneste10.rss')
        ]

        feed = feedparser.parse(RSS_feeds[self.select.value][1])

        self.afinn = Afinn(language = "da")

        # Get relevant objects from RSS feed ans store titles and scores
        for i in range(len(feed["entries"])):
            self.data_titles.append(feed["entries"][i]["title"])
            self.data_scores.append(self.afinn.score(feed["entries"][i]["title"]))

        # Dataframe
        pd.set_option('display.max_colwidth', -1) # Used to display whole title (non-truncated)
        df = pd.DataFrame({"Title": self.data_titles, "Score": self.data_scores}) # Creating the data frame and populating it

        # Highlight the positive and negative sentiments
        def highlight(s):
            if s.Score > 0:
                return ['background-color: #AAFFAA']*2
            elif s.Score < 0:
                return ['background-color: #FFAAAA']*2
            else:
                return ['background-color: #FFFFFF']*2

        df = df.style.apply(highlight, axis=1)
        display(df)
        #self.container.value = display(df)
