"""
Usage:
  Nielsen2017Linking_camera.py

Notes
-----
This script demonstrates the use of Wikidata together with
ImageNet-based deep learning classifiers. It relates to the manuscript
"Linking ImageNet WordNet Synsets with Wikidata" from 2018. 

Keras is used together with OpenCV and a pre-trained deep learning
model. The script requires the installation of at least these
component, a third-party `keras_squeezenet` package, as well as a
webcam. Internet access is required for the Danish labels as Wikidata
is queried each time the model detects a new object. The pre-trained
model used is downloaded the first time the script is run and stored
locally under `~/.keras/`.

There are several parameters in the code that might need to be
adjusted. The model can be set to, e.g., MobileNet, Resnet50 or few
other pre-trained model. The webcam that has been used had a high
resolution with a height of 1080 pixel. Depending on the webcam or
screen resolution the size and `step` parameter might need to be
changed. The language of the labels on the screen can be changed,
e.g., from 'da' to 'de' for German. 

Labels it cannot resolve are written to the terminal.

The script has run successfully under Ubuntu 17.10 with Python2 and
Python3 and tensorflow-gpu==1.4.0, Keras==2.1.4 and Cuda 8.0

Citation
--------
Finn Aarup Nielsen, Linking ImageNet WordNet Synsets with Wikidata
Wiki Workshop 2018.

Copyright
---------
Technical University of Denmark
Finn Aarup Nielsen

License
-------
Apache License, Version 2.0 
https://www.apache.org/licenses/LICENSE-2.0

Funding
-------
Innovation Foundation Denmark through the DABAI project
"""
from scipy.misc import imresize
from keras.applications import mobilenet
from keras.applications import densenet 
from keras.applications import inception_resnet_v2
import keras_squeezenet as squeezenet
from keras.preprocessing import image
from keras.applications import resnet50
import numpy as np
import cv2
try:
    from functools32 import lru_cache
except ImportError:
    from functools import lru_cache

import requests
from six import u
from time import time
from unidecode import unidecode


QUERY = """
SELECT ?item ?prefix ?synset WHERE {
  ?item wdt:P2888 ?uri .
  BIND (SUBSTR(STR(?uri), 1, 38) AS ?prefix)
  BIND (SUBSTR(STR(?uri), 39) AS ?synset)
  FILTER (?prefix = "http://wordnet-rdf.princeton.edu/wn30/")
}
"""

def synset_to_uri(synset):
    return "http://wordnet-rdf.princeton.edu/wn30/{}-n".format(synset[1:])


SYNSET_SPARQL = """
SELECT ?item ?itemLabel WHERE {{
  ?item wdt:P2888 <http://wordnet-rdf.princeton.edu/wn30/{}-n> 
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{}". }}
}}
"""
@lru_cache(maxsize=1000)
def synset_to_label(synset, language='da'):
    query = SYNSET_SPARQL.format(synset[1:], language)
    url = 'https://query.wikidata.org/sparql'
    params = {'query': query, 'format': 'json'}
    response = requests.get(url, params=params)
    data = response.json()
    labels = [item['itemLabel']['value']
              for item in data['results']['bindings']]
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


model_module = mobilenet

preprocess_input = model_module.preprocess_input
decode_predictions = model_module.decode_predictions

model_name = model_module.__name__.split('.')[-1]

if model_name == 'resnet50':
    model = model_module.ResNet50()
    model_image_size = 224
elif model_name == 'squeezenet':
    model = model_module.SqueezeNet()
    model_image_size = 227
elif model_name == 'mobilenet':
    model = model_module.MobileNet()
    model_image_size = 224
elif model_name == 'densenet':
    model = model_module.DenseNet121()
    model_image_size = 224
elif model_name == 'inceptionresnetv2':
    model = model_module.InceptionResNetV2()
    model_image_size = 299
else:
    assert False

    


font = cv2.FONT_HERSHEY_PLAIN
text_position = (10, 500)
font_scale = 1
font_color = (255, 255, 255)
line_type = 1


synset_to_label.cache_clear()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Camera view and screen size may not fit. Fullscreen is disabled for
# now.
# cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)          
# cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

capturer = cv2.VideoCapture(0)
capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = capturer.read()
#check how many pixels to skip each time
step = frame.shape[0] // model_image_size 
previous_time = 0
while(True):
    # Capture frame-by-frame
    ret, frame = capturer.read()
 
    # Preprocess image
    if False:
        x_cropped = imresize(frame, (model_image_size, model_image_size, 3))
    else:
        x_offset = (frame.shape[0] - model_image_size * step) // 2
        y_offset = (frame.shape[1] - model_image_size * step) // 2

        x_cropped = frame[x_offset:x_offset + model_image_size * step:step,
                          y_offset:y_offset + model_image_size * step:step, :]
    x = np.expand_dims(x_cropped, axis=0).astype('float32')
    x = preprocess_input(x)

    # Forward in neural network
    predictions = model.predict(x)

    # Convert predictions 
    decoded = decode_predictions(predictions)

    # Attempt to label
    label = synset_to_label(decoded[0][0][0], language='da')

    # Size of font depends on probability
    size = int(decoded[0][0][2] * 4)

    # If the label is not found some information is printed on the terminal
    if label == '???':
        message = "{} - http://image-net.org/explore.php?wnid={} - {}"
        print(message.format(synset_to_uri(decoded[0][0][0]),
                             decoded[0][0][0],
                             decoded[0][0][1]))
        label = '(' + decoded[0][0][1] + ')'
    if label.startswith('Q'):
        print("https://www.wikidata.org/wiki/" + label)
        label = '(' + decoded[0][0][1] + ')'

    # Add the label to the image    
    _ = cv2.putText(frame, unicode_to_ascii(label),
                    text_position, font, font_scale + size, (0, 0, 0), 3)
    _ = cv2.putText(frame, unicode_to_ascii(label),
                    text_position, font, font_scale + size,
                    font_color, line_type)

    # Show the image on the screen
    _ = cv2.imshow('frame', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Enable for simple benchmarking
    if False:
        now_time = time()
        print(now_time - previous_time)
        previous_time = now_time


capturer.release()
cv2.destroyAllWindows()

