from functools import lru_cache

import numpy as np
from afinn import Afinn

from src.text.utility.text_modifiers import TextModifier

_sentiment_max = 5


_sentiment_styles = {
    1: (None, None, 0.5),
    2: (None, None, 0.75),
    3: (None, None, 1.0),
    4: (None, "italic", 1.0),
    5: ("bold", "italic", 1.0),
}


__positive_test_words = "yes sweet great fantastic superb"
__negative_test_words = "no damn bad fraud prick"


@lru_cache(maxsize=2)
def _get_afinn():
    return Afinn()


def _sentiment_format(sentiment, full_contrast):
    # Get sign
    sign = np.sign(sentiment)

    # Determine formatting
    if full_contrast:
        weight = "bold"
        style = None
        color_val = 1.0
    else:
        weight, style, color_val = _sentiment_styles[int(abs(sentiment))]
    color_val = sign * color_val

    # Compute color
    color = np.array([0., 0., 0.])
    color[0] -= min(color_val, 0)
    color[2] += max(color_val, 0)

    return color, weight, style


def get_sentiment_words(text):
    # Get AFinn
    afinn = _get_afinn()

    # Get sentiment-words and their sentiments
    sentiment_words = afinn.find_all(text)
    sentiments = [afinn.score(word) for word in sentiment_words]

    return sentiments, sentiment_words


def sentiment_text_modifiers(text, full_contrast=False, lower=True):
    if lower:
        text = text.lower()

    # Get sentiments
    sentiments, sentiment_words = get_sentiment_words(text=text)

    # Make modifiers
    modifiers = []
    idx = 0
    for word, sentiment in zip(sentiment_words, sentiments):

        # Next index
        idx = text[idx:].find(word) + idx

        # End position of word
        end = idx + len(word)

        # Determine format
        color, weight, style = _sentiment_format(sentiment=sentiment, full_contrast=full_contrast)

        # Add modifier
        modifiers.append(TextModifier(idx, end, "color", color))
        if weight is not None:
            modifiers.append(TextModifier(idx, end, "weight", weight))
        if style is not None:
            modifiers.append(TextModifier(idx, end, "style", style))

        # Next
        idx = end

    return modifiers


if __name__ == "__main__":
    test_str = """Fake News Wrong forced big true.
        Fake purposely wrong, as usual!- Donal J. Trump""".strip().replace("\t", "")

    mods = sentiment_text_modifiers(text=test_str)
