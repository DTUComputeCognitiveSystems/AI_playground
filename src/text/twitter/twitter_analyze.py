from IPython.core.display import display, HTML, clear_output

from src.text.sentiment.sentiment_highlighting import sentiment_text_modifiers
from src.text.twitter.twitter_client import TwitterClient
from src.text.twitter.twitter_html import get_tweet_json, get_tweet_text, html_tweets
from src.text.utility.text_html import modified_text_to_html
from ipywidgets import Text, Button, Label, HBox, VBox, BoundedIntText


def sentiment_analyse_tweets(tweets, full_contrast=False, html_fontsize=None):
    if isinstance(tweets, (int, str)):
        tweets = [tweets]

    analyses = []
    for tweet in tweets:
        # Get tweet
        if isinstance(tweet, int):
            tweet_data = get_tweet_json(tweet_id=tweet)
        else:
            tweet_data = tweet

        # Get bare text
        tweet_text = get_tweet_text(tweet_data=tweet_data)

        # Analyse
        modifiers = sentiment_text_modifiers(text=tweet_text, full_contrast=full_contrast)

        # Make HTML
        html = modified_text_to_html(text=tweet_text, html_fontsize=html_fontsize, modifiers=modifiers)
        analyses.append(html)

    return analyses


def html_sentiment_tweets(tweet_ids, full_contrast=False, html_fontsize=2):
    if isinstance(tweet_ids, int):
        tweet_ids = [tweet_ids]

    # Get tweets' data
    tweets = [get_tweet_json(tweet_id=tweet) for tweet in tweet_ids]

    # Sentiment analysis
    analyses = sentiment_analyse_tweets(tweets=tweets, full_contrast=full_contrast, html_fontsize=html_fontsize)

    # Make HTML table with all tweets
    html = html_tweets(tweets_data=tweets, tweets_analyses=analyses)

    return html


class TwitterSentimentViewer:
    def __init__(self, twitter_client: TwitterClient, sentiment_font_size=3, full_contrast=True):
        self.full_contrast = full_contrast
        self.sentiment_font_size = sentiment_font_size
        self.twitter_client = twitter_client
        self.state = None

        self.search_text = Text(
            value='@CogSys',
            placeholder='',
            description='Search:',
            disabled=False,
            layout=dict(width="60%"),
        )
        self.search_button = Button(
            description='Search Twitter',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='',
            icon='',
            layout=dict(width="20%"),
        )
        self.description = Label(
            value="Type a search for Twitter. Use @ and # for searching for specific users or hashtags.",
            layout=dict(width="80%"),
        )
        self.n_results = BoundedIntText(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Tweets:',
            disabled=False,
            layout=dict(width="18%"),
        )

        box1 = HBox(
            (self.description, self.n_results)
        )
        box2 = HBox(
            (self.search_text, self.search_button)
        )
        self.dashboard = VBox(
            (box1, box2)
        )
        # noinspection PyTypeChecker
        display(self.dashboard)

        # Observe
        self.search_button.on_click(self._run)

    def _hold_button(self):
        self.search_button.disabled = True
        self.search_button.button_style = "warning"
        self.search_button.description = "Analysing"

    def _reset_button(self):
        self.search_button.disabled = False
        self.search_button.button_style = "success"
        self.search_button.description = "Search Twitter"

    def _run(self, _=None):
        self._hold_button()
        search_text = self.search_text.value.strip()
        n_results = self.n_results.value

        # Get tweets
        if search_text[0] == "@":
            tweets = self.twitter_client.user_timeline(username=search_text[1:], count=n_results)
        else:
            tweets = self.twitter_client.search(query=search_text, count=n_results)

        # Get twitter IDs
        tweet_ids = [tweet.id for tweet in tweets]

        # Make HTML-sentiment versions
        html = html_sentiment_tweets(tweet_ids=tweet_ids, html_fontsize=self.sentiment_font_size,
                                     full_contrast=self.full_contrast)

        self._reset_button()
        clear_output()
        # noinspection PyTypeChecker
        display(self.dashboard)
        # noinspection PyTypeChecker
        display(HTML(html))
