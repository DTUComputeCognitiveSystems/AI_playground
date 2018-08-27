import json
import os.path
import re
import urllib.parse
from calendar import month_name
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session

from src.utility.files import ensure_directory

TWITTER_API_VERSION = "1.1"
TWITTER_BASE_API_URL = "https://api.twitter.com"
TWITTER_BASE_STATUS_URL = "https://twitter.com/{}/status"
TWITTER_BASE_EMBED_URL = "https://publish.twitter.com/oembed"

TWITTER_DATE_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"
INTERNAL_DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

TWITTER_API_CALLS = {
    "search/tweets": {
        "maximum_count": 100,
        "rate_limit": 450
    },
    "statuses/user_timeline": {
        "maximum_count": 3200,
        "rate_limit": 1500
    },
    "statuses/lookup": {
        "maximum_count": 100,
        "rate_limit": 300
    }
}
TWITTER_RATE_LIMIT_WINDOW = timedelta(minutes=15)


_default_authentication_path = Path("data", "twitter", "authentication.json")
ensure_directory(_default_authentication_path)

_default_cache_directory = Path("data", "twitter", "cache")
ensure_directory(_default_cache_directory)


class TwitterClientError(Exception):
    pass


class TwitterClient:
    def __init__(self, consumer_key, consumer_secret, cache_directory=None):

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret

        self.authentication = HTTPBasicAuth(
            self.consumer_key,
            self.consumer_secret
        )

        if cache_directory is None:
            self.cache_directory = _default_cache_directory
        else:
            self.cache_directory = cache_directory

        self.session = self.open_session()

    @staticmethod
    def authenticate_from_path(path=None, cache_directory=None):
        if path is None:
            path = _default_authentication_path
        authentication_path = Path(path)

        try:
            with authentication_path.open("r") as authentication_file:
                authentication = json.loads(authentication_file.read())
            consumer_key = authentication["consumer_key"]
            consumer_secret = authentication["consumer_secret"]
        except FileNotFoundError:
            raise ValueError("No authentication found at: {}".format(path))

        return TwitterClient(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            cache_directory=cache_directory
        )

    def save_authentication_to_path(self, path=None):
        if path is None:
            path = _default_authentication_path
        authentication_path = Path(path)

        with authentication_path.open("w") as authentication_file:
            json.dump(
                dict(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret
                ),
                authentication_file
            )

    def open_session(self, access_token=None):

        client = BackendApplicationClient(client_id=self.consumer_key)

        if access_token:
            token = {
                "token_type": "bearer",
                "access_token": access_token
            }
            session = OAuth2Session(client=client, token=token)

        else:
            session = OAuth2Session(client=client)
            access_token = self.request_access_token(session)
            session = self.open_session(access_token)

        return session

    def request_access_token(self, session):

        request_token_url = build_url(TWITTER_BASE_API_URL, "oauth2/token")
        request_body = {"grant_type": "client_credentials"}

        try:
            response = session.post(
                request_token_url,
                auth=self.authentication,
                data=request_body
            )
            response_body = response.json()

            if response_body["token_type"] != "bearer":
                raise TwitterClientError(
                    "Verification of access token type failed.")

            access_token = response_body["access_token"]

        except requests.RequestException as exception:
            raise exception

        except (ValueError, KeyError):
            raise TwitterClientError("Did not receive an access token.")

        return access_token

    def search(self, query, language_code=None, count=None):

        query = re.sub("\\s{2,}", " ", query.strip())
        parameters = {"q": query}

        if language_code:
            parameters["lang"] = language_code

        tweets = self.__timeline(
            resource="search/tweets",
            parameters=parameters,
            count=count
        )

        return tweets

    def user_timeline(self, username, count=None):

        timeline = self.__timeline(
            resource="statuses/user_timeline",
            parameters={"screen_name": username},
            count=count
        )

        return timeline

    def __timeline(self, resource, parameters, count=None):

        # Setup

        if resource not in TWITTER_API_CALLS:
            raise TwitterClientError("Cannot request resource `{resource}.`")

        maximum_count = TWITTER_API_CALLS[resource]["maximum_count"]
        rate_limit = TWITTER_API_CALLS[resource]["rate_limit"]
        maximum_duration_between_retrievals = \
            TWITTER_RATE_LIMIT_WINDOW / rate_limit

        if count is None:
            count = maximum_count

        if count > maximum_count:
            print(
                "Can only retrieve up to the "
                f"{maximum_count} most recent tweets.\n"
            )
            count = maximum_count

        last_request_filename = format_filename(
            base_name="last_request",
            extension="json"
        )
        last_request_path = Path(self.cache_directory, last_request_filename)

        time_since_last_request = None
        last_request = {}

        if os.path.exists(last_request_path):
            with last_request_path.open("r") as last_request_file:
                last_request = json.load(last_request_file)
            last_request_retrieved_timestamp = datetime.strptime(
                last_request[resource]["retrieved_timestamp"],
                INTERNAL_DATE_TIME_FORMAT
            )
            now = datetime.now(timezone.utc).astimezone()
            time_since_last_request = now - last_request_retrieved_timestamp

        if time_since_last_request and \
            time_since_last_request < maximum_duration_between_retrievals:
                time_until_next_allowed_request = \
                    maximum_duration_between_retrievals \
                    - time_since_last_request
                raise TwitterClientError(
                    "Exceeded maximum number of allowed requests for now. "
                    "Wait {}.".format(format_duration(
                        time_until_next_allowed_request.total_seconds()))
                )

        # Request

        request_url = build_url(
            TWITTER_BASE_API_URL, TWITTER_API_VERSION,
            f"{resource}.json",
            tweet_mode="extended",
            count=count,
            **parameters
        )
        response = self.session.get(request_url)
        retrieved_timestamp = datetime.now(timezone.utc).astimezone()

        # TODO Handle response status codes better
        # Suspended account returns 401
        # Non-existent account return 404

        if response.status_code == 401:
            raise TwitterClientError("Bad authentication data.")

        elif response.status_code == 404:
            raise TwitterClientError("Invalid API resource.")

        elif response.status_code == 429:
            raise TwitterClientError(
                "Exceeded maximum number of allowed requests.")

        response_body = response.json()

        # Parsing

        timeline = response_body["statuses"]

        timeline = [Tweet(tweet) for tweet in timeline]

        # Saving results

        tweet_ids = [tweet.id for tweet in timeline]

        result = {
            "resource": resource,
            "parameters": parameters,
            "ids": [tweet.id for tweet in timeline],
            "retrieved_timestamp":
                retrieved_timestamp.strftime(INTERNAL_DATE_TIME_FORMAT)
        }

        timeline_filename = format_filename(
            base_name=retrieved_timestamp.strftime(INTERNAL_DATE_TIME_FORMAT)
                .replace(":", "-"),
            extension="json"
        )
        timeline_path = Path(self.cache_directory, timeline_filename)

        with timeline_path.open("w") as timeline_file:
            json.dump(
                result,
                timeline_file,
                ensure_ascii=False,
                indent=4
            )

        # Logging

        if not resource in last_request:
            last_request[resource] = {}

        last_request[resource] = result

        with open(last_request_path, "w") as last_request_file:
            json.dump(
                last_request,
                last_request_file,
                ensure_ascii=False,
                indent=4
            )

        return timeline

    def clear_cache(self):
        rmtree(self.cache_directory)


class Tweet:
    def __init__(self, raw_data):

        self.raw_data = raw_data
        self.id = self.raw_data["id"]
        self.text = self.user = self.timestamp = None
        self.hashtags = self.mentions = self.urls = None

        if "retweeted_status" in self.raw_data:
            self.__process(self.raw_data["retweeted_status"])
            self.retweet = True
            self.retweeter = User(self.raw_data["user"])
            self.retweet_timestamp = datetime.strptime(
                self.raw_data["created_at"],
                TWITTER_DATE_TIME_FORMAT
            )
        else:
            self.__process(self.raw_data)
            self.retweet = False
            self.retweeter = None
            self.retweet_timestamp = None

    @property
    def url(self):
        url = build_url(
            TWITTER_BASE_STATUS_URL.format(self.user.username),
            self.id
        )
        return url

    def text_excluding(self, hashtags=False, mentions=False, urls=False):

        text = self.text
        excluded_strings = []

        if hashtags and self.hashtags:
            excluded_strings.extend(self.hashtags)

        if mentions and self.mentions:
            excluded_strings.extend(self.mentions)

        if urls and self.urls:
            excluded_strings.extend(self.urls)

        for excluded_string in excluded_strings:
            text = text.replace(excluded_string, "")

        return text

    def as_html(self, hide_thread=False):
        request_url = build_url(
            TWITTER_BASE_EMBED_URL,
            url=self.url,
            hide_thread=hide_thread,
            dnt=True
        )
        response = requests.get(request_url)
        response_body = response.json()
        html = response_body["html"]
        return html

    def __process(self, raw_tweet):

        self.text = raw_tweet["full_text"]
        self.user = User(raw_tweet["user"])
        self.timestamp = datetime.strptime(
            raw_tweet["created_at"],
            TWITTER_DATE_TIME_FORMAT
        )

        hashtags = mentions = urls = None

        if "entities" in raw_tweet:
            entities = raw_tweet["entities"]
            if "hashtags" in entities and entities["hashtags"]:
                hashtags = entities["hashtags"]
            if "user_mentions" in entities and entities["user_mentions"]:
                mentions = entities["user_mentions"]
            if "urls" in entities and entities["urls"]:
                urls = entities["urls"]
            if "media" in entities and entities["media"]:
                if not urls:
                    urls = []
                urls.extend(entities["media"])

        if hashtags:
            self.hashtags = [f"#{hashtag['text']}" for hashtag in hashtags]

        if mentions:
            self.mentions = [f"@{mention['screen_name']}"
                             for mention in mentions]

        if urls:
            self.urls = [url["expanded_url"] for url in urls]
            for url in urls:
                self.text = self.text.replace(url["url"], url["expanded_url"])

    def __repr__(self):

        tweet_parts = [
            f"{self.user}",
            f"\"{self.text}\"",
            format_timestamp(self.timestamp)
        ]

        if self.retweet:
            tweet_parts.append(
                "Retweeted by {} on {}".format(
                    self.retweeter,
                    format_timestamp(self.retweet_timestamp)
                )
            )

        return "\n".join(tweet_parts)


class User:
    def __init__(self, raw_user):
        self.raw_user = raw_user

        self.username = self.raw_user["screen_name"]
        self.full_name = self.raw_user["name"]
        self.avatar = self.raw_user["profile_image_url_https"]

    def __repr__(self):
        return f"{self.full_name} (@{self.username})"


def build_url(base_url, *components, **parameters):
    url = base_url

    for component in components:
        component = str(component)
        url = f"{url}/{component}"

    if parameters:
        encoded_parameters = urllib.parse.urlencode(parameters)
        url = f"{url}?{encoded_parameters}"

    return url


def format_filename(base_name, extension, allow_spaces=False):

    invalid_characters = ["/", "\\", "?", "%", "*", ":", "|", "\""]

    if not allow_spaces:
        invalid_characters.append(" ")

    for character in invalid_characters:
        base_name = base_name.replace(
            character,
            urllib.parse.quote(character)
        )

    extension = extension.lstrip(".")

    filename = f"{base_name}.{extension}"

    return filename


def format_timestamp(timestamp):
    timestamp = timestamp.replace(tzinfo=timezone.utc).astimezone(tz=None)
    now = datetime.now()

    date_string = "{} {}".format(
        timestamp.day,
        month_name[timestamp.month]
    )

    if timestamp.year != now.year:
        date_string = f" {timestamp.year}"

    time_string = "{t.hour}:{t.minute:02}".format(t=timestamp)

    timestamp_string = f"{date_string} at {time_string}"

    return timestamp_string


TIME_UNITS_IN_SECONDS = {
    "year": 365*24*60*60,
    "week": 52*24*60*60,
    "day": 24*60*60,
    "hour": 60*60,
    "minute": 60,
    "second": 1,
}


def format_duration(duration_in_seconds):

    remainder_in_seconds = duration_in_seconds
    duration_string_parts = []

    for time_unit_name, time_unit_in_seconds in sorted(
            TIME_UNITS_IN_SECONDS.items(),
            key=lambda t: t[1],
            reverse=True
        ):

        duration_in_time_unit, remainder_in_seconds = divmod(
            remainder_in_seconds, time_unit_in_seconds)
        duration_in_time_unit = int(duration_in_time_unit)

        if duration_in_time_unit != 0:

            if duration_in_time_unit != 1:
                plural = "s"
            else:
                plural = ""

            duration_string_parts.append(
                f"{duration_in_time_unit} {time_unit_name}{plural}")

    if len(duration_string_parts) == 0:
        duration_string = "less than a second"
    if len(duration_string_parts) == 1:
        duration_string = duration_string_parts[0]
    if len(duration_string_parts) == 2:
        duration_string = " and ".join(duration_string_parts)
    if len(duration_string_parts) > 2:
        duration_string = ", ".join(duration_string_parts[:-1])
        duration_string += ", and " + duration_string_parts[-1]

    return duration_string
