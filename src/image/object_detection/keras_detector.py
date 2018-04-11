from collections import namedtuple
from time import time

import keras_squeezenet
import numpy as np
import pandas as pd
from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.applications import mobilenet, resnet50
from keras.applications.imagenet_utils import preprocess_input as sqnet_preprocessing, \
    decode_predictions as sqnet_decode

from src.image.models_base import ResizingImageLabeller
from src.image.video import SimpleVideo

_model_specification = namedtuple(
    "model_specificaion",
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
                         299, None, None),
    _model_specification("keras_squeezenet", keras_squeezenet, keras_squeezenet.SqueezeNet,
                         227, sqnet_preprocessing, sqnet_decode),
]


model_modules = {
    specification.name: specification for specification in _keras_models
}


class KerasDetector(ResizingImageLabeller):
    available_models = ["mobilenet", "resnet50", "densenet", "inception_resnet_v2", "keras_squeezenet"]

    def __init__(self, model_name='resnet50', resizing_method="sci_resize", verbose=False,
                 n_labels_returned=1, exlude_animals = False):

        if model_name in model_modules:
            _, self._net, model, self._input_size, self._preprocessing, self._prediction_decoding = \
                model_modules[model_name]
            self._model = model()
            self.vegetarian = exlude_animals

            if self._preprocessing is None:
                self._preprocessing = self._net.preprocess_input
            if self._prediction_decoding is None:
                self._prediction_decoding = self._net.decode_predictions
        else:
            raise ValueError("Does not know model of name: {}".format(model_name))

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
        labels = [val[1] for val in decoded]
        probabilities = [val[2] for val in decoded]

        return labels, probabilities


if __name__ == "__main__":
    print("-" * 100 + "\nSpeed Comparison of Keras Models\n" + "-" * 100 + "\n")

    # Get some frames from a video
    video = SimpleVideo(
        record_frames=True,
        frame_rate=10,
        video_length=1,
        title="Test Video"
    )
    video.start()
    frames = video.vidoe_frames

    # Go through models and time performance
    n_models = len(KerasDetector.available_models)
    times = []
    for model_nr, model_name in enumerate(KerasDetector.available_models):

        print("\n" + "-" * 40 + "\n{} / {}: {}".format(model_nr+1, n_models, model_name))

        # Make model
        model = KerasDetector(
            model_name=model_name,
            verbose=False,
            n_labels_returned=2
        )

        # Label all frames
        start_time = time()
        labels = []
        for frame in frames:
            labels.append(model.label_frame(frame=frame))
        total_time = time() - start_time
        times.append(total_time)
        print("\tTotal time   : {:.2f}s".format(total_time))
        print("\tAverage time : {:.2f}s".format(total_time / len(frames)))
        print("\tLabels       : {}".format(labels))

    # Pandas table at the end
    times = np.array(times)
    table = pd.DataFrame(
        data=np.array([times, times / len(frames)]).T,
        index=KerasDetector.available_models,
        columns=["Total time", "Average time"]
    )
    table = table.sort_values("Total time", ascending=False)
    print("\n\n" + "-" * 100)
    print("Comparison table\n")
    print(table)
