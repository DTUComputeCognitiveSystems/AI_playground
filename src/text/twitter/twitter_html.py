import re
from urllib import request
import json
import bs4
from io import BytesIO
import matplotlib.pyplot as plt
import base64

from matplotlib.figure import Figure

_PUBLISH_JSON_URL = "https://publish.twitter.com/oembed?url=https://twitter.com/Interior/status/{}"
_TABLE_FORMATTER = """
<table style="width:100%; background-color:#f1f1c1;">
    <col width="50%">
    <col width="5%">
    <col width="40%">
    {}
</table>
""".strip()
_ROW_FORMATTER = "\t" + """
    <tr style="background-color:#ffffff;">
        <td style="text-align:left">{}</td>
        <td> </td>
        <td style="text-align:left">{}</td> 
    </tr>
""".strip()


_empty_line_pattern = re.compile("<br>")


def get_tweet_json(tweet_id):
    # Get response and read
    response = request.urlopen(_PUBLISH_JSON_URL.format(tweet_id))
    tweet_data = response.read()

    # Decode and parse
    tweet_data = tweet_data.decode('utf-8')
    tweet_data = json.loads(tweet_data)

    return tweet_data


def get_tweet_text(tweet_data):
    return bs4.BeautifulSoup(tweet_data["html"], "lxml").get_text().strip()


def matplotlib2html(fig, auto_close=True):
    # Convert figure to image in memory
    bio = BytesIO()
    fig.savefig(bio, format='png')

    # Close figure
    if auto_close:
        plt.close(fig)

    # Encode figure
    encoded_fig = base64.b64encode(bio.getvalue()).decode()

    # Make HTML image tag
    image_tag = '<img src="data:image/png;base64,' + encoded_fig + '"/>'

    return image_tag


def html_tweet(tweet_data, analysis="", single_row=False):
    tweet_html = str(tweet_data["html"]) \
        .replace("<html>", "").replace("</html>", "") \
        .replace("<body>", "").replace("</body>", "")

    # For string-analyses
    if isinstance(analysis, str):
        html_row = _ROW_FORMATTER.format(tweet_html, analysis)

    # For matplotlib figures
    elif isinstance(analysis, Figure):
        html_row = _ROW_FORMATTER.format(tweet_html, matplotlib2html(fig=analysis))

    else:
        raise ValueError("Unknown analysis object in html_tweet_w_analysis")

    if single_row:
        return html_row
    return _TABLE_FORMATTER.format(html_row)


def html_tweets(tweets_data, tweets_analyses):
    rows = []

    for tweet_data, tweet_analysis in zip(tweets_data, tweets_analyses):
        rows.append(html_tweet(tweet_data=tweet_data, analysis=tweet_analysis, single_row=True))

    return _TABLE_FORMATTER.format("\n".join(rows))


if __name__ == "__main__":

    tweet_id = 996273545910145025

    test = html_tweet(get_tweet_json(tweet_id))
