import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
# import numpy as np
# import h5py
# import argparse
# import time
# import logging
from sklearn import metrics

import utilities

import keras
from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

import tensorflow as tf

def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)


# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def CRNN_audio_event_detector(model_type=None, balance_type=None, iters=50000):
    
    # Output directories
    if model_type is not None and balance_type is not None:
        model_path = os.path.join(
            'models',
            'balance_type={}'.format(balance_type),
            'model_type={}'.format(model_type)
            #'md_{}_iters.h5'.format(iters)
        )
        model_path, iters = utilities.find_model_to_load(model_path, iters)
    else:
        model_type = 'decision_level_single_attention'
        balance_type = 'balance_in_batch'
        model_path = os.path.join('models', 'md_50000_iters.h5')

    print('\n\nLoading top model with weights from:')
    print(model_path)

    time_steps = 10
    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units = 1024
    drop_rate = 0.5

    # Embedded layers
    input_layer = Input(shape=(time_steps, freq_bins))

    a1 = Dense(hidden_units)(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)

    # Pooling layers
    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.
        
        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional 
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(max_pooling, output_shape=pooling_shape)([cla])

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.
        
        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(
            average_pooling,
            output_shape=pooling_shape)(
            [cla])

    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.

        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        att = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.

        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        cla1 = Dense(classes_num, activation='sigmoid')(a2)
        att1 = Dense(classes_num, activation='softmax')(a2)
        out1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla1, att1])

        cla2 = Dense(classes_num, activation='sigmoid')(a3)
        att2 = Dense(classes_num, activation='softmax')(a3)
        out2 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla2, att2])

        b1 = Concatenate(axis=-1)([out1, out2])
        b1 = Dense(classes_num)(b1)
        output_layer = Activation('sigmoid')(b1)

    elif model_type == 'feature_level_attention':
        '''Feature level attention.

        [1] To be appear.
        '''
        cla = Dense(hidden_units, activation='linear')(a3)
        att = Dense(hidden_units, activation='sigmoid')(a3)
        b1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

        b1 = BatchNormalization()(b1)
        b1 = Activation(activation='relu')(b1)
        b1 = Dropout(drop_rate)(b1)

        output_layer = Dense(classes_num, activation='sigmoid')(b1)

    else:
        raise Exception("Incorrect model_type!")

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    # Optimization method
    learning_rate = 0.001
    optimizer = Adam(lr=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', keras.metrics.categorical_accuracy, auc_roc]
    )

    model.load_weights(model_path)

    return model
