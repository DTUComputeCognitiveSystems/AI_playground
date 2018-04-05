from functools import lru_cache
import requests
from six import u
from unidecode import unidecode

_synset_sparql_query = """
SELECT ?item ?itemLabel WHERE {{
  ?item wdt:P2888 <http://wordnet-rdf.princeton.edu/wn30/{}-n> 
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{}". }}
}}
"""

_wikidata_url = 'https://query.wikidata.org/sparql'


@lru_cache(maxsize=1000)
def synset_to_label(synset, language='en'):
    """
    Queries WordNet for the word of a specified synset in a given language.
    :param synset:
    :param language:
    :return:
    """
    # Parse to final query
    query = _synset_sparql_query.format(synset[1:], language)

    # Query wikidata and get data as JSON
    params = {'query': query, 'format': 'json'}
    response = requests.get(_wikidata_url, params=params)
    data = response.json()

    # Fetch labels
    labels = [item['itemLabel']['value']
              for item in data['results']['bindings']]

    # Return
    if len(labels) > 0:
        return labels[0]
    else:
        return "???"


def unicode_to_ascii(text):
    encoded = ''
    for character in text:
        if character == u('\xe5'):
            encoded += 'aa'
        elif character == u('\xe6'):
            encoded += 'ae'
        elif character == u('\xf8'):
            encoded += 'oe'
        elif character == u('\xf6'):
            encoded += 'oe'
        elif character == u('\xe4'):
            encoded += 'ae'
        elif character == u('\xfc'):
            encoded += 'u'
        else:
            encoded += character
    return unidecode(encoded)


if __name__ == "__main__":
    test_languages = ["en", "da", "fr", "de", "nl"]

    decoded = [[('n03207941', 'dishwasher', 0.25054157),
                ('n04442312', 'toaster', 0.240155),
                ('n04070727', 'refrigerator', 0.099175394),
                ('n04554684', 'washer', 0.065704145),
                ('n04004767', 'printer', 0.063971408)]]

    best_decoded = decoded[0][0]

    for code in test_languages:

        # Attempt to label
        unicode_label = unicode_to_ascii(synset_to_label(best_decoded[0], language=code))
        print(code + ":", unicode_label)
