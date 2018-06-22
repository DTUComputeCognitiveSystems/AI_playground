import re

from IPython.display import display
from ipywidgets import (
    Text, HTML, Label, Button, Box, HBox, VBox,
    Layout
)

from src.text.twitter.twitter_client import TwitterClient
from src.text.document_retrieval.wikipedia import Wikipedia

TWITTER_USERNAME_REGULAR_EXPRESSION = re.compile(r"\w{1,20}")

class TwitterWikipediaSearchDashboard:
    def __init__(self, twitter_client: TwitterClient,
                 wikipedia: Wikipedia):

        self.controller = TwitterWikipediaSearchController(
            dashboard=self,
            twitter_client=twitter_client,
            wikipedia=wikipedia
        )

        self.query_field = Text(
            value="",
            placeholder="Search or username",
            description="",
            disabled=False,
            layout=dict(width="auto")
        )

        self.search_with_tweets_from_user_button = Button(
            description="Search with Tweets from User",
            disabled=False,
            button_style="",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.search_with_matching_tweets_button = Button(
            description="Search with Matching Tweets",
            disabled=False,
            button_style="primary",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.search_buttons = [
            self.search_with_tweets_from_user_button,
            self.search_with_matching_tweets_button
        ]

        self.results_label = Label()
        self.results_html = HTML()

        self.search_box = Box(
            (
                self.query_field,
                self.search_with_tweets_from_user_button,
                self.search_with_matching_tweets_button
            )
        )
        
        self.results_box = VBox(
            (
                self.results_label,
                self.results_html
            )
        )
        
        self.dashboard = VBox(
            (
                self.search_box,
                self.results_box
            )
        )
        
        display(self.dashboard)
        
        self.query_field.continuous_update = False
        # self.query_field.observe(self._query_field_changed, names="value")
        
        self.search_with_matching_tweets_button.on_click(
            self._search_with_matching_tweets_button_pressed)
        self.search_with_tweets_from_user_button.on_click(
            self._search_with_tweets_from_user_button_pressed)
    
    def _disable_search_buttons(self):
        for button in self.search_buttons:
            button.disabled = True
    
    def _enable_search_buttons(self):
        for button in self.search_buttons:
            button.disabled = False
    
    def _query_field_changed(self, notification):
        if notification.type == "change":
            query = notification.new
            self.controller.search(query, twitter_method="search")

    def _search_with_matching_tweets_button_pressed(self, button):
        query = self.query_field.value
        self._disable_search_buttons()
        self.controller.search(query, twitter_method="search")
        self._enable_search_buttons()

    def _search_with_tweets_from_user_button_pressed(self, button):
        username = self.query_field.value
        self._disable_search_buttons()
        self.controller.search(username, twitter_method="user_timeline")
        self._enable_search_buttons()

    def clear_results_box(self):
        self.results_label.value = ""
        self.results_html.value = ""

    def update_results_label(self, label):
        self.results_label.value = label

    def update_results_html(self, html):
        self.results_html.value = html

    def append_to_results_html(self, html):
        self.results_html.value += "\n" + html

class TwitterWikipediaSearchController:
    def __init__(self, dashboard: TwitterWikipediaSearchDashboard,
                 twitter_client: TwitterClient,
                 wikipedia: Wikipedia):

        self.dashboard = dashboard
        self.twitter_client = twitter_client
        self.wikipedia = wikipedia

        self.content = self.tweets = self.wikipedia_results = None

    def search(self, query, twitter_method="search"):

        self.dashboard.clear_results_box()
        query = query.strip()

        if twitter_method == "search":
            label = f"Wikipedia articles related to recent tweets " \
                f"including \"{query}\"."
            self.dashboard.update_results_label(label)
            self.dashboard.append_to_results_html(
                html="<p>Downloading recent matching tweets.</p>")
            self.tweets = self.twitter_client.search(query=query)
        elif twitter_method == "user_timeline":
            is_username, username = check_twitter_username(query)
            if is_username:
                label = f"Wikipedia articles related to recent tweets " \
                    f"from @{username}."
                self.dashboard.update_results_label(label)
                self.dashboard.append_to_results_html(
                    html="<p>Downloading recent tweets from "
                        f"@{username}.</p>"
                )
                self.tweets = self.twitter_client.user_timeline(
                    username=username)
            else:
                self.content = f"\"{query}\" is not a username."

        if self.tweets:
            self.dashboard.append_to_results_html(
                html="<p>Searching local Wikipedia using tweets.</p>")
            self.wikipedia_results = self.wikipedia.search(
                query=self.tweets[0].text)
            # TODO Add number of results to results label.
            formatted_results = []
            for index, score in self.wikipedia_results[:25].items():
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
                url = document.url
                formatted_result = "\n".join([
                    # f"<article>"
                    f"<h2><a href=\"{url}\">{title}</a></h2>",
                    # f"<h1><a href=\"{url}\">{title}</a></h1>",
                    f"<p>{abstract}</p>",
                    f"<p><b>Score: {score:.3g}.</b></p>",
                    # f"</article>"
                ])
                formatted_results.append(formatted_result)
            self.content = "\n\n".join(formatted_results)

        if self.content:
            self.dashboard.update_results_html(html=self.content)


def check_twitter_username(username):

    username = username.strip()
    username = username.lstrip("@")

    if re.fullmatch(TWITTER_USERNAME_REGULAR_EXPRESSION, username):
        return True, username
    else:
        return False, username
