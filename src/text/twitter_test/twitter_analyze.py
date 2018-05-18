from src.text.sentiment.sentiment_highlighting import sentiment_text_modifiers
from src.text.twitter_test.twitter_html import get_tweet_json, get_tweet_text, html_tweets
from src.text.utility.text_html import modified_text_to_html


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


def show_tweets_with_sentiment(tweet_ids, full_contrast=False, html_fontsize=2):
    if isinstance(tweet_ids, int):
        tweet_ids = [tweet_ids]

    # Get tweets' data
    tweets = [get_tweet_json(tweet_id=tweet) for tweet in tweet_ids]

    # Sentiment analysis
    analyses = sentiment_analyse_tweets(tweets=tweets, full_contrast=full_contrast, html_fontsize=html_fontsize)

    # Make HTML table with all tweets
    html = html_tweets(tweets_data=tweets, tweets_analyses=analyses)

    return html

