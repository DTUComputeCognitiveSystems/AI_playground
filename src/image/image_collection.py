import random
from os import listdir
from os.path import isfile, join
from pathlib import Path
from shutil import rmtree

import matplotlib.image as mpimg
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from src.image.object_detection.keras_detector import KerasDetector
from src.image.video.labelled import LabelledVideo
from src.image.video.snapshot import VideoCamera
from src.real_time.matplotlib_backend import MatplotlibLoop


images_dir_name = "ml_images"


def run_video_recognition(model_name="mobilenet", video_length=10):
    """
    :param model_name: model to use for object recognition. Different models have different performances and run times
    :param video_length: length of video in seconds
    """
    net = KerasDetector(model_name=model_name, exlude_animals=True)
    the_video = LabelledVideo(net, video_length=video_length)
    the_video.start()
    while not the_video.real_time_backend.stop_now:
        plt.pause(.5)


class ImageCollector:
    def __init__(self, num_pictures=2, num_objects=2, picture_size=(224, 224)):
        """
        Used to collect images to form a dataset for ML training.
        """
        self.num_pictures = num_pictures
        self.num_objects = num_objects
        self.picture_size = picture_size

        self.frames = []
        self.labels = []
        self.net = None
        self.net_name = ""

        self.x = self.y = None
        self.loaded = False

    def run_collector(self, use_binary=False, list_of_labels=None):
        # Go through each object and collect images
        for i in range(self.num_objects):

            # Get label name
            if use_binary:
                label_name = str(bool(1 - i))
            elif list_of_labels is None:
                label_name = input("Enter label of object {}: ".format(i + 1))
            else:
                label_name = list_of_labels[i]

            # Make title
            if use_binary:
                if i:
                    title = "Images WITHOUT object"
                else:
                    title = "Images WITH object"
            else:
                title = "Object {}: {}".format(i + 1, label_name)

            # Start up the camera
            back_end = MatplotlibLoop()
            my_camera = VideoCamera(n_photos=self.num_pictures, backend=back_end, title=title,
                                    crosshair_size=self.picture_size)
            my_camera.start()

            # Wait for camera
            while not my_camera.real_time_backend.stop_now:
                plt.pause(.5)

            # Get coordinates from camera-cutout
            start_y, start_x, _, _ = my_camera.cutout_coordinates

            # Store label and cut-out-photos
            self.labels.append(label_name)
            self.frames.append(
                np.stack(my_camera.photos[:self.num_pictures])[:, start_x:start_x + self.picture_size[0],
                start_y:start_y + self.picture_size[1]])

    def load_network(self, model_name="mobilenet", net=None):
        if net is not None:
            self.net = KerasDetector(model=net)
            self.net_name = "Custom"
            return
        if self.net is None or model_name != self.net_name:
            self.net_name = model_name
            self.net = KerasDetector(model_name=self.net_name, exlude_animals=True)

    def save_images(self, file_path, use_augmentation=False):
        file_path = Path(file_path)

        if len(self.labels) == 0:
            raise IOError("No images to save")

        new_path = Path(file_path, images_dir_name)
        if new_path.exists():
            rmtree(str(new_path))

        unaugmented_path = Path(new_path, "original_imgs")
        new_path.mkdir()
        unaugmented_path.mkdir()

        # Go through labels and images
        for prefix, frame_list in zip(self.labels, self.frames):
            for i in range(len(frame_list)):
                mpimg.imsave(
                    fname=str(Path(unaugmented_path, "{}_{}.jpg".format(prefix, i))),
                    arr=frame_list[i]
                )

        # Make data-augmentation
        num_augmentations = 50 if use_augmentation else 1
        if use_augmentation:
            data_generator = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
        else:
            data_generator = ImageDataGenerator(
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0,
                horizontal_flip=False,
                fill_mode='nearest')
        for prefix, frame_list in zip(self.labels, self.frames):
            i = 0
            for _ in data_generator.flow(np.stack(frame_list), save_to_dir=new_path, save_prefix=prefix,
                                         save_format='jpg'):
                i += 1
                if i >= num_augmentations:
                    break

        # Load images for machine learning purposes
        self.load_data_from_files(file_path=new_path)

    def show_augmented(self, num_augments=4):
        """
        Visualizes a grid containing one image from each class, together with a number of data-augmentations.
        :param int num_augments: Number of augmentations for each image.
        """

        # Get data-generator
        data_generator = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=20,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Add some augmented frames from each label
        augmented_frames = []
        for i in range(len(self.labels)):
            # Get a random image from class
            frame = random.choice(self.frames[i])

            # Add original frame
            augmented_frames.append([frame])
            img_iterator = data_generator.flow(frame[np.newaxis, :])

            # Add augmentations
            # TODO image type not correct
            for j in range(num_augments):
                augmented_frames[i].append(img_iterator.next()[0])

        # Prepare figure and axes-grid
        plt.close("all")
        num_objects = len(self.labels)
        f, ax_array = plt.subplots(num_objects, num_augments + 1, figsize=(4 * (num_augments + 1), 3 * num_objects))

        # Make image plots for each object
        for i in range(num_objects):
            for j in range(num_augments + 1):
                # Set labelling
                if j == 0 and i == 0:
                    ax_array[i, j].set_title("True image")
                    ax_array[i, j].set_ylabel(self.labels[i])
                elif i == 0:
                    ax_array[i, j].set_title("Augmented")

                # Show image
                ax_array[i, j].imshow(augmented_frames[i][j])
                ax_array[i, j].tick_params(which='both', labelbottom=False, labelleft=False, length=0)

    def show_images(self, model_name="mobilenet", net=None):
        # Load network for classification
        self.load_network(model_name=model_name, net=net)

        # Prepare figure and axes
        plt.close("all")
        num_objects = len(self.labels)
        f, ax_array = plt.subplots(num_objects, self.num_pictures, figsize=(4 * self.num_pictures, 3 * num_objects))
        ax_array = np.asarray(ax_array) if num_objects == 1 and self.num_pictures == 1 else ax_array
        ax_array = np.expand_dims(ax_array, axis=0) if num_objects == 1 else ax_array
        ax_array = np.expand_dims(ax_array, axis=-1) if self.num_pictures == 1 else ax_array

        # Go through objects
        for i in range(num_objects):
            print(100 * "_" + "\nObject {}".format(self.labels[i]))
            for j in range(self.num_pictures):
                # Set labels
                if j == 0:
                    ax_array[i, j].set_ylabel(self.labels[i], rotation=0, size='large',
                                              labelpad=5 * (len(self.labels[i])))
                if i == 0:
                    ax_array[i, j].set_title("Picture {}".format(j))

                # Get the most probable label from classifier
                probable_labels = self.net.label_frame(self.frames[i][j])[0]
                print("Picture {}: ".format(j), ", ".join(probable_labels))

                # Show image
                ax_array[i, j].imshow(self.frames[i][j])
                ax_array[i, j].tick_params(which='both', labelbottom=False, labelleft=False, length=0)

    def load_data_from_files(self, file_path):
        # Ensure format
        file_path = Path(file_path)

        # Get files and split into categories
        files = sorted(list(file_path.glob("*.jpg")))
        positive_img_files = [f for f in files if "True" in f.name]
        negative_img_files = [f for f in files if "False" in f.name]

        # Load images
        try:
            pos_imgs = np.stack(mpimg.imread(str(file_name)).astype(np.float64) for file_name in positive_img_files)
            neg_imgs = np.stack(mpimg.imread(str(file_name)).astype(np.float64) for file_name in negative_img_files)

            # Stack images and labels
            self.x = np.vstack([pos_imgs, neg_imgs])
            self.y = np.hstack([np.ones((len(pos_imgs))), np.zeros((len(neg_imgs)))])

            self.loaded = True

        except ValueError:
            self.loaded = False

    def load_ml_data(self, random_permutation=True, train_split=0.9):
        if self.x is None:
            raise ValueError("No test-data has been loaded. Either load from file or take some photos.")

        # Get data
        x = self.x
        y = self.y

        # Do a random permutation of data
        if random_permutation:
            permutation = np.random.permutation(len(y))
            x = x[permutation]
            y = y[permutation]

        # Preprocess inputs for keras
        x_proc = preprocess_input(x)

        # Split into training and validation
        n_train = int(train_split * len(x))
        x_train = x_proc[:n_train]
        y_train = y[:n_train]
        x_val = x_proc[n_train:]
        y_val = y[n_train:]

        return (x_train, y_train), (x_val, y_val)


# def load_data(file_path, random_permutation=True, train_split=0.9):
#     file_path = Path(file_path)
#
#     # Get files and split into categories
#     files = sorted(list(file_path.glob("*.jpg")))
#     positive_img_files = [f for f in files if "True" in f.name]
#     negative_img_files = [f for f in files if "False" in f.name]
#
#     # Load images
#     pos_imgs = np.stack(mpimg.imread(str(file_name)).astype(np.float64) for file_name in positive_img_files)
#     neg_imgs = np.stack(mpimg.imread(str(file_name)).astype(np.float64) for file_name in negative_img_files)
#
#     # Stack images and labels
#     x = np.vstack([pos_imgs, neg_imgs])
#     y = np.hstack([np.ones((len(pos_imgs))), np.zeros((len(neg_imgs)))])
#
#     # Do a random permutation of data
#     if random_permutation:
#         permutation = np.random.permutation(len(y))
#         x = x[permutation]
#         y = y[permutation]
#
#     # Preprocess inputs for keras
#     x_proc = preprocess_input(x)
#
#     # Split into training and validation
#     n_train = int(train_split * len(x))
#     x_train = x_proc[:n_train]
#     y_train = y[:n_train]
#     x_val = x_proc[n_train:]
#     y_val = y[n_train:]
#
#     return (x_train, y_train), (x_val, y_val)
