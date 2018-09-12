import os
import glob
import time

import pyaudio
import librosa
import numpy as np
import wave
import IPython.display
from scipy import signal
import sklearn.preprocessing
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D

class Recorder:
    
    def __init__(self, n_classes=2, n_files=12, prefix='aib', wav_dir=None):
    
        if wav_dir is None:
            self.wav_dir = "/tmp"
        else:
            self.wav_dir = wav_dir

        self.n_classes = n_classes
        self.n_files = n_files
        self.prefix = prefix

        # sampling rate, use 22050 to match pre trained model
        self.sample_rate = 22050
    
    def record(self, seconds=4, framesize=2205):
        
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                 # Must be Mono 
        RATE = self.sample_rate      # sampling rate (Hz), 22050 was used for this application
        FRAMESIZE = framesize        # buffer size, number of data points to read at a time
        RECORD_SECONDS = seconds     # how long should the recording (approx) be
        NOFRAMES = int((RATE * RECORD_SECONDS) / FRAMESIZE)  # number of frames needed

        try:
            for i in range(self.n_classes):

                IPython.display.clear_output(wait=True)
                print('Get ready to record class {}, {} files'.format(i, self.n_files))
                time.sleep(3)
                IPython.display.clear_output(wait=True)
                
                for cnt in [3, 2, 1]:
                    print('Get ready to record class {}, {} files'.format(i, self.n_files))
                    print('{}...'.format(cnt))
                    time.sleep(1)
                    IPython.display.clear_output(wait=True)                

                # instantiate pyaudio
                p = pyaudio.PyAudio()

                # open stream
                stream = p.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=FRAMESIZE)

                for j in range(self.n_files):
                    
                    IPython.display.clear_output(wait=True)
                    print("Recording class {}, file no {}/{}...".format(i, j + 1, self.n_files))

                    data = stream.read(NOFRAMES * FRAMESIZE)
                    decoded = np.frombuffer(data, dtype=np.int16)

                    print("Finished recording, writing file...")

                    fname = self.prefix + "{0:d}-{1:03d}.wav".format(i, j + 1)
                    print(fname)

                    wavefile = wave.open(os.path.join(self.wav_dir, fname), 'wb')
                    wavefile.setnchannels(CHANNELS)
                    wavefile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wavefile.setframerate(RATE)
                    wavefile.writeframes(data)

                    wavefile.close()
                    
                stream.stop_stream()
                stream.close()
                p.terminate()

        except KeyboardInterrupt:  # (this is Kernel -> Interrupt in a notebook")
            stream.stop_stream()
            stream.close()
            p.terminate()
        
    def get_files(self):
        file_ext = self.prefix + "*.wav"
        files = glob.glob(os.path.join(self.wav_dir, file_ext))
        return files
              
    def create_dataset(self):
        """ Create a labeled dataset """
        
        data = []
        labels = []
        
        # get files
        files = self.get_files()
        
        for fn in files:
            # get the label
            label = fn.split(self.prefix)[1].split('-')[0]
            
            # load the file at the desired sampling rate. Loads as dtype=np.float32
            (sound_clip, fs) = librosa.load(fn, sr=self.sample_rate)

            data.append(sound_clip)
            labels.append(label)
    
        data = np.array(data)

        labels = np.array(labels, dtype=np.int)        
        n_labels = len(labels)
        one_hot_encode = np.zeros((n_labels, self.n_classes))
        one_hot_encode[np.arange(n_labels), labels] = 1
        labels = one_hot_encode
        
        return data, labels
        

class SoundClassifier:
    
    def __init__(self, weights_path='sound_classification_weights.hdf5', mix=False):
        
        self.weights_path = weights_path
        self.mix = mix
        
        ### Recording settings ###
        
        # sampling rate, use 22050 to match pre trained model
        self.sample_rate = 22050
        
        ### Preprocessing settings  ###
        
        # number of frequency bands 
        self.n_bands = 150
        # number of frames
        self.n_frames = 150
        # number of channels in spectrogram image
        self.n_channels = 3
        
        # window length for spectrogram, 
        # should match 150 frames in 4 seconds with 50% overlap
        self.n_window = int(self.sample_rate * 4. / self.n_frames * 2) - 4 * 2
        # 50% overlap
        self.n_overlap = int(self.n_window / 2.)
        # filter to be used for log-mel transformation of spectrogram
        self.melW = librosa.filters.mel(sr=self.sample_rate, 
                                   n_fft=self.n_window, 
                                   n_mels=self.n_bands, 
                                   fmin=0., fmax=8000.)
        # Hamming window
        self.ham_win = np.hamming(self.n_window)
        
        # build and load pre-trained model
        self.base_model = self._build_base_model()
        
    def train(self, data=None, wav_dir=None):
        """
        :param data: data and labels in a list or tuple, if wav_dir is None
        :param wav_dir: directory containing wav files, if data is None
        """
        
        # Get data from input or directory
        if wav_dir is not None:
            self.sound_clips, self.labels = self._create_dataset(wav_dir)
        else:
            self.sound_clips, self.labels = data
            
        # number of classes
        self.n_classes = self.labels.shape[1]
        
        # Extract features
        self.features = self._preprocess()
        
        # Scale frequencies across sound clips
        self._scaling()
        
        # mix classes
        if self.mix:
            self._mix_classes()
        
        # Split data into train, val and test set
        self._split()
        
        # Send the input through the base model
        self.base_train = self.base_model.predict(self.train_data)
        self.base_val = self.base_model.predict(self.validation_data)
        self.base_test = self.base_model.predict(self.test_data)
        
        # build and compile top model
        self.top_model = self._build_top_model()
        
        # train the top model
        self._train_top_model()
        
    def _build_base_model(self):
        model = Sequential()

        # section 1

        model.add(Convolution2D(filters=32, kernel_size=5,
                                strides=2,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal",
                                input_shape=(self.n_frames, self.n_bands, self.n_channels)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=32, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.3))

        # section 2    
        model.add(Convolution2D(filters=64, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=64, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # section 3
        model.add(Convolution2D(filters=128, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=128, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=128, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=128, kernel_size=3,
                                strides=1,
                                padding="same",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # section 4
        model.add(Convolution2D(filters=512, kernel_size=3,
                                strides=1,
                                padding="valid",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(filters=512, kernel_size=1,
                                strides=1,
                                padding="valid",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # section 5
        model.add(Convolution2D(filters=10, kernel_size=1,
                                strides=1,
                                padding="valid",
                                kernel_regularizer=l2(0.0001),
                                kernel_initializer="normal"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())

        model.add(Activation('softmax'))
        
        # load the saved checkpoint weights
        # model.load_weights('/Users/nbip/proj/dtu/dtu-bach/dev/sound_classification_weights.hdf5')
        model.load_weights(self.weights_path)
        
        # pop the top layers?
        # Try to remove a different number of layers. Removing 2 seems to work okay
        for _ in range(2):
            model.layers.pop()

        # needed fix, to have an output of the model
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model
        
    def _preprocess(self):
        """ Most of the pre-processing, except scaling """
        
        log_specgrams_list = []
        
        for sound_clip in self.sound_clips:
            log_specgram = self._process_sound_clip(sound_clip)
            log_specgrams_list.append(log_specgram)
    
        return np.array(log_specgrams_list)
    
    def _process_sound_clip(self, sound_clip=None, fn=None):
        
        if fn is not None:
            (sound_clip, fs) = librosa.load(fn, sr=self.sample_rate)

        # Compute spectrogram                
        [f, t, x] = signal.spectral.spectrogram(
            x=sound_clip,
            window=self.ham_win,
            nperseg=self.n_window,
            noverlap=self.n_overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')

        # Log mel transform
        x = np.dot(x.T, self.melW.T)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32).T

        # make sure the number of timesteps is n_time
        if x.shape[1] < self.n_frames:
            pad_shape = (x.shape[0], self.n_frames - x.shape[1])
            pad = np.ones(pad_shape) * np.log(1e-8)
            #x_new = np.concatenate((x, pad), axis=1)
            x = np.hstack((x, pad))
        # no pad necessary - truncate
        else:
            x = x[:, 0: self.n_frames]
            
        # make list into an array, (n_freqs, n_time, 1)
        log_specgram = x.reshape(self.n_bands, self.n_frames, 1)
        # extend the last dimension to 3 channels, 
        features = np.concatenate((log_specgram, np.zeros(np.shape(log_specgram))), axis=2)
        features = np.concatenate((features, np.zeros(np.shape(log_specgram))), axis=2)

        # add something to channel 2 and 3
        # first order difference, computed over 9-step window
        features[:, :, 1] = librosa.feature.delta(features[:, :, 0])
        # for using 3 dimensional array to use ResNet and other frameworks
        features[:, :, 2] = librosa.feature.delta(features[:, :, 1])

        features = np.array(features)
        features = np.transpose(features, (1, 0, 2))
        
        return features 
    
    def _scaling(self):
        """ 
        Each frequency band should have mean 0 and std 1 across the training data,
        this provides a better classification accuracy
        """
        
        # for each channel, compute scaling factor
        self.scaler_list = []
        (n_clips, n_time, n_freq, n_channel) = self.features.shape

        for channel in range(n_channel):
            xtrain_2d = self.features[:, :, :, channel].reshape((n_clips * n_time, n_freq))
            scaler = sklearn.preprocessing.StandardScaler().fit(xtrain_2d) # standardization of features
            # print("Channel %d Mean: %s" % (channel, scaler.mean_,))
            # print("Channel %d Std: %s" % (channel, scaler.scale_,))
            # print("Calculating scaler time: %s" % (time.time() - t1,))
            self.scaler_list += [scaler]
            
        
        self.features = self._do_scaling(self.features)
        
    def _do_scaling(self, data):
        x4d_scaled = np.zeros(data.shape)
        (n_clips, n_time, n_freq, n_channel) = data.shape

        for channel in range(n_channel):
            x2d = data[:, :, :, channel].reshape((n_clips * n_time, n_freq))
            x2d_scaled = self.scaler_list[channel].transform(x2d)
            x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
            x4d_scaled[:, :, :, channel] = x3d_scaled

        return x4d_scaled  
        
    def _get_files(self, wav_dir):
        file_ext = "*.wav"
        files = glob.glob(os.path.join(wav_dir, file_ext))
        return files
              
    def _create_dataset(self, wav_dir):
        """ Create a labeled dataset """
        
        data = []
        labels = []
        
        # get files
        files = self._get_files(wav_dir)
        
        for fn in files:
            # get the label
            label = fn.split(self.prefix)[1].split('-')[0]
            
            # load the file at the desired sampling rate. Loads as dtype=np.float32
            (sound_clip, fs) = librosa.load(fn, sr=self.sample_rate)

            data.append(sound_clip)
            labels.append(label)
    
        data = np.array(data)

        labels = np.array(labels, dtype=np.int)        
        n_labels = len(labels)
        one_hot_encode = np.zeros((n_labels, self.n_classes))
        one_hot_encode[np.arange(n_labels), labels] = 1
        labels = one_hot_encode
        
        return data, labels
        
    def _split(self):
        """ Split data into train, val and test set"""
        
        # number of observations
        n = self.features.shape[0]

        # intertwine the two classes, so every second is class 0 and every other second is class 1
        mixed_x = np.empty(self.features.shape, dtype=self.features.dtype)
        mixed_x[0::2] = self.features[:int(n/2), :]
        mixed_x[1::2] = self.features[int(n/2):, :]

        mixed_y = np.empty(self.labels.shape, dtype=self.labels.dtype)
        mixed_y[0::2] = self.labels[:int(n/2), :]
        mixed_y[1::2] = self.labels[int(n/2):, :]

        self.train_data = mixed_x[:int(0.8 * n), :]
        self.train_labels = mixed_y[:int(0.8 * n), :]
        self.validation_data = mixed_x[int(0.8 * n): int(0.9 * n), :]
        self.validation_labels = mixed_y[int(0.8 * n): int(0.9 * n), :]
        self.test_data = mixed_x[int(0.9 * n):, :]
        self.test_labels = mixed_y[int(0.9 * n):, :]

        print('train {}, val {}, test {}'.format(len(self.train_data), len(self.validation_data), len(self.test_data)))
        
    def _mix_classes(self):
        """
        [1] Tokozume Y, Ushiku Y, Harada T. Learning from Between-class Examples for Deep Sound Recognition. arXiv preprint arXiv:1711.10282. 2017 Nov 28.
        """
        (n, height, width, depth) = self.features.shape
        
        # How many extra data points do we want?
        n_extra = 5 * n
        
        # create randomly mixed examples
        idx = np.argmax(self.labels, axis=1)
        
        extra_data = np.zeros((n_extra, height, width, depth))
        extra_labels = np.zeros((n_extra, self.n_classes))
            
        for i in range(n_extra):
            # choose two observations from different classes
            c = range(self.n_classes)
            c0, c1 = np.random.choice(c, 2, replace=False)
            idx0 = idx == c0
            idx1 = idx == c1
            
            dat0 = self.features[idx0, :]
            dat1 = self.features[idx1, :]
            
            obs0 = np.random.choice(len(dat0), 1)
            obs1 = np.random.choice(len(dat1), 1)
            x0 = dat0[obs0, :]
            x1 = dat1[obs1, :]
            
            # random mixing coefficient
            r0 = np.random.rand()
            r1 = 1 - r0
            
            # sound pressure in the two examples, use RMS a A-weighting
            G0 = np.sqrt(np.mean(x0**2))
            G1 = np.sqrt(np.mean(x1**2))
            
            # From equation 2 of [1]
            p = 1 / (1 + 10**((G0 - G1)/20) * (r1/r0) )            
            # obtain the mixed sample
            mixed_sample = (p*x0 + (1 - p)*x1) / np.sqrt(p**2 + (1 - p)**2)
            
            extra_data[i, :] = mixed_sample 
            extra_labels[i, c0] = r0   # r0
            extra_labels[i, c1] = r1   # r1

        self.features = extra_data  #np.vstack((self.features, extra_data))
        self.labels = extra_labels  #np.vstack((self.labels, extra_labels))
       
    def _build_top_model(self):
        # add a new dense top layers
        fcmodel = Sequential()
        fcmodel.add(Flatten(input_shape=self.base_train.shape[1:]))
        # fcmodel.add(Dense(256, activation='relu', input_shape=[10]))
        fcmodel.add(Dense(256, activation='relu'))
        fcmodel.add(Dropout(0.5))
        fcmodel.add(Dense(self.n_classes, activation='sigmoid'))
        # Multilabel classification, therefore sigmoid and not softmax
        # Therefore also binary_crossentropy instead of multiclass_crossentropy

        if self.mix:
            fcmodel.compile(optimizer='rmsprop', loss='binary_crossentropy')
        else:
            fcmodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        return fcmodel
    
    def _train_top_model(self):
        self.history = self.top_model.fit(self.base_train, self.train_labels,
                              epochs=40,
                              batch_size=1,
                              validation_data=(self.base_val, self.validation_labels))
        
        y_prob = self.top_model.predict(self.base_test)
        y_pred = np.argmax(y_prob, axis=1)
        
        labels = np.argmax(self.test_labels, axis=1)

        print('Test accuracy: ', np.mean(y_pred == labels))
        
        
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        
        # summarise history for training and validation set loss
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax1.set_title('Model Loss')
        ax1.set_ylabel('loss', fontsize = 'large')
        ax1.set_xlabel('epoch', fontsize = 'large' )
        ax1.legend(['train', 'validation'], loc='upper left')

        if not self.mix:
            # summarise history for training and validation set accuracy
            ax2.plot(self.history.history['acc'])
            ax2.plot(self.history.history['val_acc'])
            ax2.set_title('Model Accuracy')
            ax2.set_ylabel('accuracy')
            ax2.set_xlabel('epoch')
            ax2.legend(['train', 'validation'], loc='upper left')
            
        plt.show()

    
    def predict(self, sound_clip=None, fn=None):
        
        # preprocess file or soundcip
        pre_processed = self._process_sound_clip(sound_clip=sound_clip, fn=fn)
        
        # scaling
        scaled = self._do_scaling(pre_processed[np.newaxis, :, :, :])
        
        # feed through base model
        base = self.base_model.predict(scaled)
        
        # do the prediction
        pred = self.top_model.predict(base)
        
        return pred
        

