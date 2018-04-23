# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:02:54 2018

@author: lauri
"""
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.losses import categorical_crossentropy
from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled import LabelledVideo
class Classifier:
    def __init__(self):
        self.model= Sequential()
        self.model.add(MaxPooling2D(pool_size=(4, 4),input_shape=(224, 224, 3)))
        
        self.model.add(Flatten( ))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss=categorical_crossentropy,
                      optimizer=keras.optimizers.Adagrad(),
                      metrics=['accuracy'])
        
    def train(self,train_data, num_epochs = 5 ):
        x_train, y_train = train_data
        self.model.fit(x_train, y_train,
          batch_size=128,
          epochs=num_epochs,
          verbose=1,
          validation_split=0.1)
    def evaluate(self,test_data):
        x_val, y_val = test_data
        loss, acc = self.model.evaluate(x_val, y_val)
        print("Accuracy on the validation set is {:}%.".format(100*acc))
    def run_live(self,video_length = 10):
        the_video = LabelledVideo(KerasDetector(model = self.model),video_length=video_length )
        the_video.start()
        while(not the_video.real_time_backend.stop_now ):
            plt.pause(.5)

        

        
        