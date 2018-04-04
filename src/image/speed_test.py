

from keras.applications import mobilenet, resnet50
from keras.applications import densenet 
from keras.applications import inception_resnet_v2
import keras_squeezenet 

from keras.preprocessing import image
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




    
step = 3

font = cv2.FONT_HERSHEY_PLAIN
text_position = (10, 500)
font_scale = 1
font_color = (255, 255, 255)
line_type = 1



# Camera view and screen size may not fit. Fullscreen is disabled for
# now.
# cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)          
# cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

capturer = cv2.VideoCapture(0)
capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
num_tests = 10
_, frame = capturer.read()

#%%
height, width, depth =frame.shape

x_list = np.empty((num_tests, height, width,depth))
for i in range(num_tests):
    _, x_list[i] =  capturer.read()
capturer.release()
cv2.destroyAllWindows()
#%%
from scipy.misc import imresize
#%%
times = {}
for net_name in [ mobilenet, resnet50, densenet,inception_resnet_v2, keras_squeezenet]:
    
    model_module = net_name
    

    
    model_name = model_module.__name__.split('.')[-1]

    if model_name == "keras_squeezenet":
        from keras.applications.imagenet_utils import preprocess_input, decode_predictions
    else:
        preprocess_input = model_module.preprocess_input
        decode_predictions = model_module.decode_predictions
        
    if model_name == 'resnet50':
        model = model_module.ResNet50()
        model_image_size = 224
    elif model_name == 'keras_squeezenet':
        model = model_module.SqueezeNet()
        model_image_size = 227
    elif model_name == 'mobilenet':
        model = model_module.MobileNet()
        model_image_size = 224
    elif model_name == 'densenet':
        model = model_module.DenseNet121()
        model_image_size = 224
    elif model_name == 'inception_resnet_v2':
        model = model_module.InceptionResNetV2()
        model_image_size = 299
    else:
        assert False

    
    previous_time = time()
    for frame in x_list:
        # Capture frame-by-frame
        
        
    
        # Preprocess image
        x_cropped = imresize(frame, (model_image_size, model_image_size, 3))
        x = np.expand_dims(x_cropped, axis=0).astype('float32')
#        x_offset = (frame.shape[0] - model_image_size * step) / 2
#        y_offset = (frame.shape[1] - model_image_size * step) / 2
#        x_cropped = frame[x_offset:x_offset + model_image_size * step:step,
#                          y_offset:y_offset + model_image_size * step:step, :]
#        x = np.expand_dims(x_cropped, axis=0).astype('float32')
#        x = preprocess_input(x)
    
        # Forward in neural network
        predictions = model.predict(x)

    
        # Convert predictions 
        decoded = decode_predictions(predictions)
    

    
        # Enable for simple benchmarking
    now_time = time()
    times[model_name] = (now_time-previous_time)/ num_tests
    print("Avg time per image for ", model_name,": ",  times[model_name])
    del model

    

import json

with open("speed_logs","w")  as file:


    file.write(json.dumps(times)) 
