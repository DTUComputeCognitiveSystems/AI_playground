import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, Activation
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.losses import binary_crossentropy
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np

from src.image.object_detection.keras_detector import KerasDetector
from keras.applications.imagenet_utils import preprocess_input
from src.image.video.labelled_video import LabelledVideo


class SimpleBinaryClassifier:
    def __init__(self):
        """
        Simple class to wrap a predefined neural network
        """
        self.model = Sequential()
        self.model.add(AveragePooling2D(pool_size=(4, 4), input_shape=(224, 224, 3)))
        self.model.add(Conv2D(16, (9, 9)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (5, 5)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss=binary_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def _decoding(self, x):
        # Get probability of first label
        prob = x[0][0]

        return [[[
            "", "True {:.4f}".format(prob) if prob > 0.5 else "False {:.4f}".format(prob),
            prob if prob > 0.5 else 1 - prob
        ]]]

    def train(self, train_data, num_epochs=5, verbose=0, callbacks=None):
        """
        :param train_data:
        :param num_epochs:
        :param verbose:
        :type callbacks: object
        """
        if callbacks is None:
            callbacks = []
        x_train, y_train = train_data  # type: np.ndarray, np.ndarray
        x_train = preprocess_input(np.copy(x_train))

        self.model.fit(x_train, y_train,
                       batch_size=512,
                       epochs=num_epochs,
                       verbose=verbose,
                       validation_split=0.1, callbacks=callbacks)

    def evaluate(self, test_data):
        x_val, y_val = test_data
        x_val = preprocess_input(np.copy(x_val))
        loss, acc = self.model.evaluate(x_val, y_val)
        print("Accuracy on the validation set is {:}%.".format(100 * acc))

    def run_live(self, video_length=10):
        the_video = LabelledVideo(
            KerasDetector(
                model_specification=self.model,
                preprocessing=lambda x: preprocess_input(np.copy(x).astype(np.float32)),
                decoding=self._decoding
            ),
            video_length=video_length,
            crosshair_type='box',
            crosshair_size=(224, 224),
        )
        the_video.start()
        while not the_video.real_time_backend.stop_now:
            plt.pause(.5)
