import re

from IPython.display import display, HTML as IPHTML
from ipywidgets import (
    interact,
    Text, HTML, Label,
    BoundedIntText, BoundedFloatText,
    Button,
    Output,
    Box, VBox, HBox
)
from matplotlib import pyplot

from src.text.twitter.twitter_client import TwitterClient
from src.text.document_retrieval.wikipedia import Wikipedia

TWITTER_USERNAME_REGULAR_EXPRESSION = re.compile(r"\w{1,20}")

DEFAULT_NUMBER_OF_TWEETS = 10
DEFAULT_TERM_FREQUENCY_SCALING_PARAMETER_VALUE = 1.2
DEFAULT_DOCUMENT_LENGTH_SCALING_PARAMETER_VALUE = 0.75
DEFAULT_RESULTS_WIDGET = HTML("<i>None.</i>")


class TwittipediaController:
    def __init__(self, twitter_client: TwitterClient,
                 wikipedia: Wikipedia):

        self.view = TwittipediaView(controller=self)
        self.twitter_client = twitter_client
        self.wikipedia = wikipedia

        self.content = self.tweets = self.wikipedia_results = None
    
    def search_tweets(self, query, twitter_method="search", count=None):

        # TODO Clear results

        query = query.strip()

        if twitter_method == "search":
            self.tweets = self.twitter_client.search(
                query=query,
                language=self.wikipedia.language_code,
                count=count
            )
        elif twitter_method == "user_timeline":
            is_username, username = check_twitter_username(query)
            if is_username:
                self.tweets = self.twitter_client.user_timeline(
                    username=username,
                    count=count
                )

        if self.tweets:
            self.view.show_tweets(self.tweets)

    def search_wikipedia(self, k_1=1.2, b=0.75):

        if self.tweets:
            for i, tweet in enumerate(self.tweets):
                wikipedia_results, search_terms = self.wikipedia.search(
                    query=tweet.text_excluding(urls=True),
                    k_1=k_1,
                    b=b
                )
                formatted_result_list = ["<ol>"]
                for index, score in wikipedia_results[:5].items():
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
                self.view.show_articles_for_tweet_number(i, formatted_results)


class TwittipediaView:
    def __init__(self, controller: TwittipediaController):

        self.controller = controller

        title = HTML(
            value="<h1>Twittipedia</h1>",
            description="",
            disabled=False
        )

        # Twitter

        twitter_title = HTML(
            value="<b>Tweets</b>",
            description="",
            disabled=False
        )

        self.query_field = Text(
            value="",
            placeholder="Search or enter username",
            description="",
            disabled=False,
            layout=dict(width="auto")
        )
        self.query_field.observe(self._query_field_changed, names="value")

        self.number_of_tweets_field = BoundedIntText(
            value=DEFAULT_NUMBER_OF_TWEETS,
            min=1,
            max=100,
            step=1,
            description="",
            disabled=False,
            layout=dict(width="auto")
        )
        self.number_of_tweets_field.observe(
            self._number_of_tweets_field_changed,
            names="value"
        )
        number_of_tweets_label = Label(
            value="most recent tweets",
            disabled=False,
            layout=dict(width="auto")
        )
        number_of_tweets_field_with_label = HBox(
            (
                self.number_of_tweets_field,
                number_of_tweets_label
            )
        )

        self.search_tweets_button = Button(
            description="Search Tweets",
            disabled=True,
            button_style="primary",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.load_tweets_from_user_button = Button(
            description="Load Tweets from User",
            disabled=True,
            button_style="",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.twitter_search_buttons = [
            self.load_tweets_from_user_button,
            self.search_tweets_button
        ]

        for button in self.twitter_search_buttons:
            button.on_click(self._twitter_search_button_pressed)

        twitter_search_buttons_box = Box(
            self.twitter_search_buttons,
            layout=dict(
                justify_content="flex-end",
                flex_flow="row wrap",
            )
        )

        self.reset_and_clear_tweets_button = Button(
            description="Reset and Clear Tweets",
            disabled=True,
            button_style="danger",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.reset_and_clear_tweets_button.on_click(
            self._reset_and_clear_tweets_button_pressed)

        self.twitter_buttons = self.twitter_search_buttons \
            + [self.reset_and_clear_tweets_button]
        twitter_buttons_box = Box(
            (
                self.reset_and_clear_tweets_button,
                twitter_search_buttons_box
            ),
            layout=dict(
                justify_content="space-between",
                flex_flow="row wrap",
            )
        )

        twitter_box = VBox(
            (
                twitter_title,
                self.query_field,
                number_of_tweets_field_with_label,
                twitter_buttons_box
            )
        )

        # Wikipedia search

        wikipedia_title = HTML(
            value="<b>Wikipedia Search</b>",
            description="",
            disabled=False
        )

        self.wikipedia_options = []

        self.term_frequency_scaling_parameter_field = BoundedFloatText(
            value=DEFAULT_TERM_FREQUENCY_SCALING_PARAMETER_VALUE,
            min=0,
            max=100,
            step=0.1,
            description="",
            disabled=False,
            layout=dict(width="auto")
        )
        self.term_frequency_scaling_parameter_field.observe(
            self._term_frequency_scaling_parameter_field_changed,
            names="value"
        )
        self.wikipedia_options.append({
            "label": "$k_1$${{}}=$",
            "field": self.term_frequency_scaling_parameter_field,
            "explanation": "$k_1 \ge 0$"
        })

        self.document_length_scaling_parameter_field = BoundedFloatText(
            value=DEFAULT_DOCUMENT_LENGTH_SCALING_PARAMETER_VALUE,
            min=0,
            max=1,
            step=0.01,
            description="",
            disabled=False,
            layout=dict(width="auto")
        )
        self.document_length_scaling_parameter_field.observe(
            self._document_length_scaling_parameter_field_changed,
            names="value"
        )
        self.wikipedia_options.append({
            "label": "$b$${{}}=$",
            "field": self.document_length_scaling_parameter_field,
            "explanation": "$0 \le b$${} \le 1$"
        })

        wikipedia_options_box = Box(
            (
                VBox(
                    [
                        Label(value=option["label"])
                        for option in self.wikipedia_options
                    ],
                    layout=dict(
                        align_items="flex-end"
                    )
                ),
                VBox(
                    [option["field"] for option in self.wikipedia_options],
                ),
                VBox(
                    [
                        Label(value="(" + option["explanation"] + ")")
                        for option in self.wikipedia_options
                    ]
                )
            )
        )

        self.search_wikipedia_button = Button(
            description="Search Wikipedia with Tweets",
            disabled=True,
            button_style="primary",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.search_wikipedia_button.on_click(
            self._search_wikipedia_button_pressed)

        self.reset_and_clear_wikipedia_results_button = Button(
            description="Reset and Clear Results",
            disabled=True,
            button_style="danger",
            tooltip="",
            icon="",
            layout=dict(width="auto")
        )
        self.reset_and_clear_wikipedia_results_button.on_click(
            self._reset_and_clear_wikipedia_results_button_pressed)

        self.wikipedia_buttons = [
            self.reset_and_clear_wikipedia_results_button,
            self.search_wikipedia_button
        ]

        wikipedia_buttons_box = Box(
            self.wikipedia_buttons,
            layout=dict(
                justify_content="space-between",
                flex_flow="row wrap",
            )
        )

        wikipedia_box = VBox(
            (
                wikipedia_title,
                wikipedia_options_box,
                wikipedia_buttons_box
            )
        )

        # Result views

        results_title = HTML(
            value="<b>Results</b>",
            description="",
            disabled=True
        )

        self.results_box = VBox([DEFAULT_RESULTS_WIDGET])
        self.results = None

        # Together

        self.buttons = self.twitter_buttons + self.wikipedia_buttons

        search_box = VBox(
            (
                twitter_box,
                wikipedia_box
            ),
            # layout=dict(max_width="600px")
        )

        results_box = VBox(
            (
                results_title,
                self.results_box
            )
        )

        self.widget = VBox(
            (
                title,
                search_box,
                results_box
            )
        )

        self._reset_and_clear_tweets()

    def _disable_twitter_buttons(self):
        for button in self.twitter_buttons:
            button.disabled = True

    def _enable_twitter_buttons(self):
        for button in self.twitter_buttons:
            button.disabled = False

    def _disable_twitter_search_buttons(self):
        for button in self.twitter_search_buttons:
            button.disabled = True

    def _enable_twitter_search_buttons(self):
        for button in self.twitter_search_buttons:
            button.disabled = False

    def _disable_wikipedia_buttons(self):
        for button in self.wikipedia_buttons:
            button.disabled = True

    def _enable_wikipedia_buttons(self):
        for button in self.wikipedia_buttons:
            button.disabled = False

    def _disable_all_buttons(self):
        for button in self.buttons:
            button.disabled = True

    def _enable_all_buttons(self):
        for button in self.buttons:
            button.disabled = False

    def _reset_and_clear_tweets(self):
        self.query_field.value = ""
        self.number_of_tweets_field.value = \
            DEFAULT_NUMBER_OF_TWEETS
        self.results_box.children = [DEFAULT_RESULTS_WIDGET]
        self._disable_twitter_buttons()
        self.search_wikipedia_button.disabled = True
        self._reset_and_clear_wikipedia_results()

    def _reset_and_clear_wikipedia_results(self):
        self.term_frequency_scaling_parameter_field.value = \
            DEFAULT_TERM_FREQUENCY_SCALING_PARAMETER_VALUE
        self.document_length_scaling_parameter_field.value = \
            DEFAULT_DOCUMENT_LENGTH_SCALING_PARAMETER_VALUE
        self._clear_wikipedia_results()
        self.reset_and_clear_wikipedia_results_button.disabled = True

    def _clear_wikipedia_results(self):
        if isinstance(self.results, list):
            for result in self.results:
                if "articles" in result:
                    result["articles"].value = ""

    def _query_field_changed(self, notification):

        if notification.type == "change":

            query = notification.new

            if query:
                self.reset_and_clear_tweets_button.disabled = False
                self.search_tweets_button.disabled = False
            else:
                self.reset_and_clear_tweets_button.disabled = True
                self.search_tweets_button.disabled = True

            is_username, username = check_twitter_username(query)

            if is_username:
                self.load_tweets_from_user_button.disabled = False
            else:
                self.load_tweets_from_user_button.disabled = True

    def _number_of_tweets_field_changed(self, notification):

        if notification.type == "change":

            number_of_tweets = notification.new

            if number_of_tweets == DEFAULT_NUMBER_OF_TWEETS:
                self.reset_and_clear_tweets_button.disabled = True
            else:
                self.reset_and_clear_tweets_button.disabled = False

    def _term_frequency_scaling_parameter_field_changed(self, notification):

        if notification.type == "change":

            term_frequency_scaling_parameter = notification.new

            if term_frequency_scaling_parameter == \
                DEFAULT_TERM_FREQUENCY_SCALING_PARAMETER_VALUE:

                self.reset_and_clear_wikipedia_results_button.disabled = True
            else:
                self.reset_and_clear_wikipedia_results_button.disabled = False

    def _document_length_scaling_parameter_field_changed(self, notification):

        if notification.type == "change":

            document_length_scaling_parameter = notification.new

            if document_length_scaling_parameter == \
                DEFAULT_DOCUMENT_LENGTH_SCALING_PARAMETER_VALUE:

                self.reset_and_clear_wikipedia_results_button.disabled = True
            else:
                self.reset_and_clear_wikipedia_results_button.disabled = False

    def _twitter_search_button_pressed(self, button):

        query = self.query_field.value
        count = self.number_of_tweets_field.value

        if button is self.search_tweets_button:
            twitter_method = "search"
            query_string = f"Searching for recent tweets matching \"{query}\"..."

        elif button is self.load_tweets_from_user_button:
            twitter_method = "user_timeline"
            username = query.lstrip("@")
            query_string = f"Loading tweets from @{username}..."

        self.results_box.children = [HTML(f"<i>{query_string}</i>")]

        self._disable_all_buttons()

        self.controller.search_tweets(
            query=query,
            twitter_method=twitter_method,
            count=count
        )

        self._enable_twitter_buttons()

    def _reset_and_clear_tweets_button_pressed(self, button):
        self._reset_and_clear_tweets()

    def _reset_and_clear_wikipedia_results_button_pressed(self, button):
        self._reset_and_clear_wikipedia_results()

    def _search_wikipedia_button_pressed(self, button):

        k_1 = self.term_frequency_scaling_parameter_field.value
        b = self.document_length_scaling_parameter_field.value

        self._disable_all_buttons()

        self.controller.search_wikipedia(
            k_1=k_1,
            b=b
        )

        self._enable_all_buttons()

    def show_tweets(self, tweets):

        self.results = []

        for tweet in tweets:

            result = {
                "tweet": Output(
                    layout=dict(width="50%")
                ),
                "articles": HTML(
                    layout=dict(width="50%")
                )
            }
            self.results.append(result)

            with result["tweet"]:
                display(IPHTML(tweet.as_html(hide_thread=True)))

        self.results_box.children = [
            HBox(
                (
                    result["tweet"],
                    result["articles"]
                )
            )
            for result in self.results
        ]

        display(IPHTML('<script id="twitter-wjs" type="text/javascript" async defer src="//platform.twitter.com/widgets.js"></script>'))
        self.search_wikipedia_button.disabled = False

    def show_articles_for_tweet_number(self, i, formatted_results):
        if isinstance(self.results, list) \
            and i < len(self.results) \
            and "articles" in self.results[i] \
            and isinstance(self.results[i]["articles"], HTML):
                self.results[i]["articles"].value = formatted_results

    # def clear_results_box(self):
    #     self.results_label.value = ""
    #     self.results_html.value = ""
    #
    # def update_results_label(self, label):
    #     self.results_label.value = label
    #
    # def update_results_html(self, html):
    #     self.results_html.value = html
    #
    # def append_to_results_html(self, html):
    #     self.results_html.value += "\n" + html


class Twittipedia(TwittipediaController):
    def __init__(self, twitter_client: TwitterClient,
                 wikipedia: Wikipedia):
        super(Twittipedia, self).__init__(
            twitter_client = twitter_client,
            wikipedia = wikipedia
        )
        display(self.view.widget)

def check_twitter_username(username):

    username = username.strip()
    if username.startswith("@"):
        username = username.replace("@", "", 1)

    if re.fullmatch(TWITTER_USERNAME_REGULAR_EXPRESSION, username):
        return True, username
    else:
        return False, username
