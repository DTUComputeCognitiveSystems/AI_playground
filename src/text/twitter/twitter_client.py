import json
import os.path
import re
import urllib.parse
from calendar import month_name
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session

from src.utility.files import ensure_directory

TWITTER_API_VERSION = "1.1"
TWITTER_BASE_API_URL = "https://api.twitter.com"

TWITTER_DATE_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"
INTERNAL_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"

TWITTER_API_CALLS = {
    "search/tweets": {
        "maximum_count": 100,
        "rate_limit": 450
    },
    "statuses/user_timeline": {
        "maximum_count": 3200,
        "rate_limit": 1500
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

    def search(self, query, count=100):

        query = re.sub("\\s{2,}", " ", query.strip())

        tweets = self.__timeline(
            resource="search/tweets",
            parameters={"q": query},
            count=count
        )

        return tweets

    def user_timeline(self, username, count=100):

        timeline = self.__timeline(
            resource="statuses/user_timeline",
            parameters={"screen_name": username},
            count=count
        )

        return timeline

    def __timeline(self, resource, parameters, count):

        if resource not in TWITTER_API_CALLS:
            raise TwitterClientError("Cannot request resource `{resource}.`")

        maximum_count = TWITTER_API_CALLS[resource]["maximum_count"]
        rate_limit = TWITTER_API_CALLS[resource]["rate_limit"]
        maximum_duration_between_retrievals = \
            TWITTER_RATE_LIMIT_WINDOW / rate_limit

        if count > maximum_count:
            print(
                "Can only retrieve up to the "
                f"{maximum_count} most recent tweets.\n"
            )
            count = maximum_count

        timeline_directory = Path(
            self.cache_directory,
            resource.replace("/", "-")
        )
        ensure_directory(timeline_directory)

        timeline_filename = format_filename(
            base_name="-".join([f"{k}={v}" for k, v in parameters.items()]),
            extension="json"
        )
        timeline_path = Path(timeline_directory, timeline_filename)

        timeline = None
        time_since_retrieval = None

        if os.path.exists(timeline_path):
            with timeline_path.open("r") as timeline_file:
                response_body = json.load(timeline_file)
            retrieved_timestamp = datetime.strptime(
                response_body["retrieved_timestamp"],
                INTERNAL_DATE_TIME_FORMAT
            )
            now = datetime.now(timezone.utc)
            time_since_retrieval = now - retrieved_timestamp
            timeline = response_body["statuses"]

        if (not timeline
                or time_since_retrieval > maximum_duration_between_retrievals
                or len(timeline) < count):

            request_url = build_url(
                TWITTER_BASE_API_URL, TWITTER_API_VERSION,
                f"{resource}.json",
                tweet_mode="extended",
                count=count,
                **parameters
            )
            response = self.session.get(request_url)
            retrieved_timestamp = datetime.now(timezone.utc)

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

            if isinstance(response_body, list):
                response_body = {"statuses": response_body}

            response_body["retrieved_timestamp"] = datetime.strftime(
                retrieved_timestamp,
                INTERNAL_DATE_TIME_FORMAT
            )

            if response_body["statuses"]:
                with timeline_path.open("w") as timeline_file:
                    json.dump(response_body, timeline_file)

            timeline = response_body["statuses"]

        timeline = [Tweet(tweet) for tweet in timeline[:count]]

        return timeline


class Tweet:
    def __init__(self, raw_data):

        self.raw_data = raw_data
        self.id = self.text = self.user = self.timestamp = None

        if "retweeted_status" in self.raw_data:
            self.process(self.raw_data["retweeted_status"])
            self.retweet = True
            self.retweeter = User(self.raw_data["user"])
            self.retweet_timestamp = datetime.strptime(
                self.raw_data["created_at"],
                TWITTER_DATE_TIME_FORMAT
            )
        else:
            self.process(self.raw_data)
            self.retweet = False
            self.retweeter = None
            self.retweet_timestamp = None

    def process(self, some_raw_data):
        self.id = self.raw_data["id"]
        self.text = parse_tweet_text(
            some_raw_data["full_text"],
            some_raw_data["entities"]
        )
        self.user = User(some_raw_data["user"])
        self.timestamp = datetime.strptime(
            some_raw_data["created_at"],
            TWITTER_DATE_TIME_FORMAT
        )

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


def parse_tweet_text(text, entities=None):
    urls = []

    if entities:
        if "urls" in entities and entities["urls"]:
            urls.extend(entities["urls"])
        if "media" in entities and entities["media"]:
            urls.extend(entities["media"])

    for url in urls:
        text = text.replace(url["url"], url["display_url"])

    return text


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
