from collections import namedtuple
from time import time

import keras
import numpy as np
import pandas as pd
import json
from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.applications import mobilenet, resnet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

from src.image.object_detection.models_base import ResizingImageLabeller
#from src.image.video import SimpleVideo

_model_specification = namedtuple(
    "model_specification",
    "name, net, model, input_size, preprocessing, prediction_decoding"
)

_keras_models = [
    _model_specification("mobilenet", mobilenet, mobilenet.MobileNet,
                         224, None, None),
    _model_specification("resnet50", resnet50, resnet50.ResNet50,
                         224, None, None),
    _model_specification("densenet", densenet, densenet.DenseNet121,
                         224, None, None),
    _model_specification("inception_resnet_v2", inception_resnet_v2, inception_resnet_v2.InceptionResNetV2,
                         299, None, None)
]

model_modules = {
    specification.name: specification for specification in _keras_models
}


class KerasDetector(ResizingImageLabeller):
    available_models = ["mobilenet", "resnet50", "densenet", "inception_resnet_v2"]

    def __init__(self,
                 model_specification='resnet50', resizing_method="sci_resize",
                 preprocessing=None, decoding=lambda x: [x],
                 n_labels_returned=1, exlude_animals=False, verbose=False, language="eng-US"):
        self._preprocessing = preprocessing
        self._prediction_decoding = decoding
        self.vegetarian = exlude_animals
        self.language = language # 3-letter language codes used in ISO 639-2, plus locale in 2 letters
        self.language_labels_list = None # List of labels for chosen language
        self.language_loaded = False # Used to determine, what source to use for the labels [eng-US, dan-DK]

        # Load the language file
        try:
            json_data=open("imagenet_class_index_dk.json").read()
            self.language_labels_list = json.loads(json_data)
            self.language_loaded = True
        except:
            raise Exception("No language file found for Keras. Continuing with eng-US")
        finally:
            pass

        # Get model by name
        if isinstance(model_specification, str):
            model_name = model_specification

            # Go through models to find the matching one
            if model_name in model_modules:
                _, self._net, model, self._input_size, self._preprocessing, self._prediction_decoding = \
                    model_modules[model_name]
                self._model = model()  # type: resnet50.ResNet50

                if self._preprocessing is None:
                    self._preprocessing = self._net.preprocess_input
                if self._prediction_decoding is None:
                    self._prediction_decoding = self._net.decode_predictions
            else:
                raise ValueError("Does not know model of name: {}".format(model_name))

        # Otherwise set to specifications in arguments
        else:
            self._model = model_specification  # type: resnet50.ResNet50
            self._input_size = self._model.input_shape[1]

            # Ensure that all parts are given
            assert self._prediction_decoding is not None and self._preprocessing is not None, \
                "You must specify a preprocessing and prediction_decoding step for this model."

        # Parameters for reshaping
        self._model_input_shape = (self._input_size, self._input_size, 3)
        super().__init__(model_input_shape=self._model_input_shape, resizing_method=resizing_method,
                         n_labels_returned=n_labels_returned, verbose=verbose)

    def _label_frame(self, frame):
        """
        :param np.ndarray frame:
        :return:
        """

        # Expand batch-dimension
        frame = np.expand_dims(frame, axis=0).astype('float32')

        # Use networks preprocessing
        frame = self._preprocessing(frame)

        # Forward in neural network
        predictions = self._model.predict(x=frame)
        if self.vegetarian:
            predictions[0, :397] = 0
        
        # Convert predictions (index for removing batch-dimension)
        decoded = self._prediction_decoding(predictions)[0]

        # Get labels and probabilities
        if self.language_loaded == True and self.language == "dan-DK":
            k = [key for key, value in self.language_labels_list.items() if decoded[0][0] == value[0]]
            #print("k: {}, d: {}, l: {}".format(k, decoded[0][0], self.language_labels_list[k[0]][2]))
            labels = [self.language_labels_list[k[0]][2]]
        elif self.language_loaded == True and self.language == "eng-US":
            k = [key for key, value in self.language_labels_list.items() if decoded[0][0] == value[0]]
            labels = [self.language_labels_list[k[0]][1]]
        else:
            labels = [val[1] for val in decoded]

        probabilities = [val[2] for val in decoded]

        return labels, probabilities


def configure_simple_model():
    """ loads a pretrained model and replaces the last layer with a layer to finetune"""
    base_model = densenet.DenseNet121(input_shape=(224, 224, 3), include_top=False)

    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='softmax')(x)

    # create graph of your new model
    head_model = Model(input=base_model.input, output=predictions)

    head_model.compile(loss=keras.losses.binary_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])
    return head_model


# if __name__ == "__main__":
#     print("-" * 100 + "\nSpeed Comparison of Keras Models\n" + "-" * 100 + "\n")

#     # Get some frames from a video
#     video = SimpleVideo(
#         record_frames=True,
#         frame_rate=10,
#         video_length=1,
#         title="Test Video"
#     )
#     video.start()
#     frames = video.video_frames

#     # Go through models and time performance
#     n_models = len(KerasDetector.available_models)
#     times = []
#     for model_nr, a_model_name in enumerate(KerasDetector.available_models):

#         print("\n" + "-" * 40 + "\n{} / {}: {}".format(model_nr + 1, n_models, a_model_name))

#         # Make model
#         model = KerasDetector(
#             model_specification=a_model_name,
#             verbose=False,
#             n_labels_returned=2
#         )

#         # Label all frames
#         start_time = time()
#         labels = []
#         for frame in frames:
#             labels.append(model.label_frame(frame=frame))
#         total_time = time() - start_time
#         times.append(total_time)
#         print("\tTotal time   : {:.2f}s".format(total_time))
#         print("\tAverage time : {:.2f}s".format(total_time / len(frames)))
#         print("\tLabels       : {}".format(labels))

#     # Pandas table at the end
#     times = np.array(times)
#     table = pd.DataFrame(
#         data=np.array([times, times / len(frames)]).T,
#         index=KerasDetector.available_models,
#         columns=["Total time", "Average time"]
#     )
#     table = table.sort_values("Total time", ascending=False)
#     print("\n\n" + "-" * 100)
#     print("Comparison table\n")
#     print(table)
