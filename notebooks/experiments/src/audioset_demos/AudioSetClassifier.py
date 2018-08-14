

# third-party sounds processing and visualization library
import librosa
import vggish_input
#import vggish_params
import vggish_postprocess
import vggish_keras

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_weights.ckpt'
pca_params_path = 'vggish_pca_params.npz'

import numpy as np
import os

# signal processing library
from scipy import signal
from scipy.io import wavfile
# import six
# import tensorflow as tf
# import h5py

import audio_event_detection_model as AEDM
import utilities

from download_youtube_wav import download_youtube_wav

from VGGish_base_model import VGGish

class AudioSetClassifier:
    
    def __init__(self,
        model_type=None,
        balance_type=None,
        iters=50000,
        mix=False):
        
        self.mix = mix
        
        ### Recording settings ###
        
        # number of classes
        self.n_classes = 527
        # number of frequency bands 
        self.n_bands = 64
        # number of frames
        self.n_frames = 10
        # number of channels in spectrogram image
        self.n_channels = 1
        # sampling rate, use 22050 to match pre trained model
        self.sample_rate = 16000
        
        ### Preprocessing settings  ###
        
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
        
        self.model_type = model_type
        self.balance_type = balance_type
        self.iters = iters

        # build and load pre-trained model
        print('\nLoading VGGish base model:')
        self.base_model = self._build_base_model()

        self.top_model = self._build_top_model()
        # have to initialize before threading
        self.base_model._make_predict_function()
        self.top_model._make_predict_function()

        # Postprocess the results to produce whitened quantized embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params_path)
        
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
        
        # mix classes
#         if self.mix:
#             self._mix_classes()
        
        # build and compile top model
        self.top_model = self._build_top_model()
        
        # train the top model
        self._train_top_model()
        
    def _build_base_model(self):
        # Define VGGish, load the checkpoint, and run the batch through the model to
        # produce embeddings.
        model = vggish_keras.get_vggish_keras()
        model.load_weights(checkpoint_path)
        model.summary()
        return model
        #return VGGish()
        
    def _preprocess(self):
        """ Most of the pre-processing, except scaling """
    

        log_specgrams_list = []
        
        for sound_clip in self.sound_clips:
            log_specgram = self._process_sound_clip(sound_clip)
            log_specgrams_list.append(log_specgram)
    
        return np.array(log_specgrams_list)
    

    def _process_sound_clip(self, sound_clip=None, fn=None, sample_rate=16000):
        
        # assert sound_clip.dtype == np.int16, 'Bad sample type: %r' % sound_clip.dtype
        # samples = sound_clip / 32768.0  # Convert to [-1.0, +1.0]

        if isinstance(sound_clip, str):# and '.wav' in sound_clip:
            proc_sound_clip = vggish_input.wavfile_to_examples(sound_clip)
        elif isinstance(sound_clip, np.ndarray):
            proc_sound_clip = vggish_input.waveform_to_examples(
                sound_clip,
                sample_rate
            )
        else:
            print("Warning: The sound clip does not have the right format")
        
        return np.expand_dims(proc_sound_clip, -1)
    
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
        Tokozume Y, Ushiku Y, Harada T. Learning from Between-class Examples for Deep Sound Recognition. arXiv preprint arXiv:1711.10282. 2017 Nov 28.
        """
        (n, height, width, depth) = self.features.shape
        
        # How many extra data points do we want?
        n_extra = 4 * n
        
        # create randomly mixed examples
        idx = np.argmax(self.labels, axis=1)
        
        extra_data = np.zeros((n_extra, height, width, depth))
        extra_labels = np.zeros((n_extra, self.n_classes))
        for i in range(n_extra):
            obs0 = np.random.choice(len(idx), 1)
            obs1 = np.random.choice(len(idx), 1)
            # r0 = np.round(np.random.rand())
            # r1 = np.round(np.random.rand())
            r0 = 0.5 + np.random.rand()
            r1 = 0.5 + np.random.rand()
            mixed_sample = r0 * self.features[obs0, :] + r1 * self.features[obs1, :]
            extra_data[i, :] = mixed_sample / (r0 + r1 + 1e-8)
            extra_labels[i, np.argmax(self.labels[obs0, :])] = r0   # r0
            extra_labels[i, np.argmax(self.labels[obs1, :])] = r1   # r1
        self.features = np.vstack((self.features, extra_data))
        self.labels = np.vstack((self.labels, extra_labels))
               
#     def _mix_classes(self):
#         (n, height, width, depth) = self.base_train.shape
        
#         # How many extra data points do we want?
#         n_extra = 4 * n
        
#         # create randomly mixed examples
#         idx = np.argmax(self.train_labels, axis=1)
        
#         extra_data = np.zeros((n_extra, height, width, depth))
#         extra_labels = np.zeros((n_extra, self.n_classes))
#         for i in range(n_extra):
#             obs0 = np.random.choice(len(idx), 1)
#             obs1 = np.random.choice(len(idx), 1)
#             # r0 = np.round(np.random.rand())
#             # r1 = np.round(np.random.rand())
#             r0 = np.random.rand()
#             r1 = np.random.rand()
#             mixed_sample = r0 * self.base_train[obs0, :] + r1 * self.base_train[obs1, :]
#             extra_data[i, :] = mixed_sample / (r0 + r1 + 1e-8)
#             extra_labels[i, np.argmax(self.train_labels[obs0, :])] = r0   # r0
#             extra_labels[i, np.argmax(self.train_labels[obs1, :])] = r1   # r1
#         self.base_train = np.vstack((self.base_train, extra_data))
#         self.train_labels = np.vstack((self.train_labels, extra_labels))
       
    def _build_top_model(self):
        model = AEDM.CRNN_audio_event_detector(
            model_type=self.model_type,
            balance_type=self.balance_type,
            iters=self.iters
        )

        return model
    
    def _train_top_model(self):
        self.history = self.top_model.fit(self.base_train, self.train_labels,
                              epochs=40,
                              batch_size=1,
                              validation_data=(self.base_val, self.validation_labels))
        
        y_prob = self.top_model.predict(self.base_test)
        y_pred = np.round(y_prob).squeeze()
        print(y_pred)
        print(self.test_labels)
        print('Test accuracy: ', np.mean(y_pred == self.test_labels))
        
        
    def plot_training(self):
        fig = plt.figure(figsize=(16, 8))
        print("History keys:", (self.history.history.keys()))
        # summarise history for training and validation set accuracy
        plt.subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # summarise history for training and validation set loss
        plt.subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss', fontsize = 'large')
        plt.xlabel('epoch', fontsize = 'large' )
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def embed(self, sound_clip, fn=None, sample_rate=16000):
        # preprocess file or soundclip
        pre_processed = self._process_sound_clip(
            sound_clip=sound_clip,
            fn=fn,
            sample_rate=sample_rate
        )
        
        # print(pre_processed.shape)

        # feed through base model
        base = self.base_model.predict(pre_processed)
        
        # print(base.shape)

        postprocessed_batch = self.pproc.postprocess(base)

        return postprocessed_batch

    def predict(self, sound_clip, fn=None, sample_rate=16000):
        postprocessed_batch = self.embed(
            sound_clip,
            fn=fn,
            sample_rate=sample_rate
        )

        batch_shape = postprocessed_batch.shape

        if batch_shape[0] > 10:
            postprocessed_batch = postprocessed_batch[:10]
        elif batch_shape[0] < 10:
            zero_pad = np.zeros((self.n_frames, batch_shape[1]))
            zero_pad[:batch_shape[0]] = postprocessed_batch
            postprocessed_batch = zero_pad

        # x_tr_pad = np.zeros(
        #     shape=(self.n_frames, batch_shape[1])
        # )

        # pad_ind = min(batch_shape[0], self.n_frames)

        # x_tr_pad[:pad_ind] = postprocessed_batch[:pad_ind]

        # print(postprocessed_batch.shape)

        (x_tr, y_tr) = utilities.transform_data(postprocessed_batch)

        # print(batch_shape)

        # pad = vggish_params.NUM_FRAMES - x_tr.shape[0]

        # # x_tr = np.pad(
        # #     x_tr, 
        # #     pad_width=(
        # #             (0, pad),
        # #             (0, 0)
        # #     ),
        # #     mode='constant'
        # # )

        # do the prediction
        pred = self.top_model.predict(np.expand_dims(x_tr, 0))
        
        return pred
        