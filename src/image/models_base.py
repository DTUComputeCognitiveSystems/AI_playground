from collections import Callable

import numpy as np
from scipy.misc import imresize


def _scipy_resize(frame, resizing_method, model_input_shape):
    for method in ["nearest", "lanczos", "bilinear", "bicubic" or "cubic"]:
        if method in resizing_method:
            new_frame = imresize(frame, model_input_shape, interp=method)
            break
    else:
        new_frame = imresize(frame, model_input_shape)

    return new_frame


class ImageLabeller:
    def __init__(self, model_input_shape, resizing_method="sci_resize",
                 verbose=False):
        """
        :param str | Callable resizing_method:
        :param tuple | list | np.ndarray model_input_shape:
        """
        self._verbose = verbose
        self._model_input_shape = model_input_shape

        if resizing_method is None:
            self._resizing_method = lambda x: x

        elif isinstance(resizing_method, Callable):
            self._vprint("Received Callable resizing method.")
            self._resizing_method = resizing_method

        elif "sci_resize" in resizing_method:
            self._vprint("Using scipy resizing.")
            self._resizing_method = lambda x: _scipy_resize(frame=x,
                                                            resizing_method=resizing_method,
                                                            model_input_shape=model_input_shape)

        else:
            raise ValueError("Does not understand resizing_method: {}".format(resizing_method))

    def _vprint(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def _label_frame(self, frame):
        raise NotImplementedError

    def label_frame(self, frame):
        """
        :param np.ndarray frame:
        :return:
        """
        self._vprint("Original frame-size: {}".format(frame.shape))
        new_frame = self._preprocess_frame(frame=frame)
        self._vprint("Preprocessed frame-size: {}".format(frame.shape))
        label = self._label_frame(frame=new_frame)
        return label

    def _preprocess_frame(self, frame):
        frame = self._resizing_method(frame)
        return frame
