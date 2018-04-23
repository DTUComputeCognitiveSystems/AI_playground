# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:33:12 2018

@author: lauri
"""
from src.image.video.labelled import LabelledVideo
from src.real_time.matplotlib_backend import MatplotlibLoop
from src.image.video.snapshot import VideoCamera
from matplotlib import pyplot as plt
from src.image.object_detection.keras_detector import KerasDetector
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from os.path import isfile, join
import matplotlib.image as mpimg
from os import listdir
from keras.utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
def run_video_recognition(model_name = "mobilenet", video_length = 10):

    net = KerasDetector(model_name=model_name,exlude_animals=  True)
    the_video = LabelledVideo(net,video_length=video_length)
    the_video.start()
    while(not the_video.real_time_backend.stop_now ):
        plt.pause(.5)
class Image_Collector:
    def __init__(self, num_pictures =2, num_objects =2):
        self.num_pictures = num_pictures
        self.num_objects = num_objects

        self.frames = []
        self.labels = []
        self.net = None
        self.net_name = ""
    def run_collector(self, use_binary = False):
        if use_binary:
            instructions = ["Hold object before camera to take picture and press enter", 
                "Take pictures without the object by pressing enter"]
        else:
            instructions = ["Hold object before camera to take picture and press enter" for i in range(self.num_objects)]
        for i in range(self.num_objects):
            if use_binary:
                label_name = str(bool(1-i))
            else:
                label_name = input("Object label: ")
            
            
            back_end = MatplotlibLoop()
            my_camera = VideoCamera(n_photos = self.num_pictures, backend=back_end, title = instructions[i])
            
        
            my_camera.start()
            
            while(not my_camera.real_time_backend.stop_now ):
                plt.pause(.5)
            self.start_coordinates, _ =my_camera._cross_hair.coordinates
            self.labels.append(label_name)
            self.frames.append(my_camera.photos[:self.num_pictures])
            my_camera._finalize()
    def load_network(self,model_name = "mobilenet"):
        
        if self.net == None or model_name != self.net_name:
            self.net_name = model_name
            self.net = KerasDetector(model_name="mobilenet",exlude_animals=  True)
    
    def save_images(self,filepath,use_augmentation = False):
        if len(self.labels)==0:
            raise IOError("No images to save")
        num_augmentations = 30 if use_augmentation else 1
        (start_x, start_y) = self.start_coordinates
        if use_augmentation:
            datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        else:
            datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            fill_mode='nearest')
        for prefix, cur_list in zip(self.labels, self.frames):
            i = 0
            for batch in datagen.flow(np.stack(cur_list)[:, start_x:start_x+224, start_y:start_y+224],save_to_dir=filepath, save_prefix=prefix, save_format='jpg'):
                i += 1
                if i >= num_augmentations:
                    break 

        
            
            
            
    def show_images(self, model_name = "mobilenet"):
        self.load_network(model_name = model_name)
        plt.close("all")
        num_objects= len(self.labels)
        f, axarr = plt.subplots(num_objects, self.num_pictures, figsize=(4*self.num_pictures,3*num_objects))
        axarr=np.asarray(axarr) if num_objects ==1 and self.num_pictures==1 else axarr
        axarr = np.expand_dims(axarr, axis=0) if num_objects ==1 else axarr
        axarr =  np.expand_dims(axarr, axis=-1) if self.num_pictures ==1 else axarr
        
        for i in range (num_objects):
            print(100*"_" +"\nObject {}".format(self.labels[i]) )
            for j in range(self.num_pictures):
                if j==0:
                    axarr[i,j].set_ylabel(self.labels[i], rotation=0, size='large', labelpad=5*(len(self.labels[i])))
                axarr[i,j].set_title("Picture {}".format(j))
                probable_labels=self.net.label_frame(self.frames[i][j])[0]
                print("Picture {}: ".format(j),", ".join(probable_labels))
                axarr[i,j].imshow(self.frames[i][j])
                axarr[i,j].tick_params( which='both', labelbottom=False,labelleft = False,length=0)
def load_data(filepath):
    positive_img_files = [f for f in listdir(filepath) if isfile(join(filepath, f)) and "True" in f]
    negative_img_files = [f for f in listdir(filepath) if isfile(join(filepath, f)) and "False" in f]
    positive_imgs = np.stack (mpimg.imread(join(filepath, file_name)).astype(np.float64) for file_name in positive_img_files)
    neg_imgs = np.stack (mpimg.imread(join(filepath, file_name)).astype(np.float64) for file_name in negative_img_files)
    x = np.vstack([positive_imgs, neg_imgs])
    y = to_categorical(np.hstack([np.ones((len(positive_imgs))), np.zeros((len(neg_imgs)))]), num_classes= 2)
    random_permuation = np.random.permutation(len(y))
    x = x[random_permuation]
    y= y[random_permuation]
    x_proc= preprocess_input(x)
    x_train = x_proc[:int(0.9*len(x))]
    y_train = y[:int(0.9*len(x))]

    x_val = x_proc[int(0.9*len(x)):]
    y_val = y[int(0.9*len(x)):]
    return (x_train, y_train), (x_val, y_val)