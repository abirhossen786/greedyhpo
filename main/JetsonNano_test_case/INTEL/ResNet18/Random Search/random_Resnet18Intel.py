from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from timeit import default_timer as timer
from datetime import timedelta

import tensorflow as tf

MEMORY_LIMIT = 700
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.initializers import glorot_uniform
import pickle
from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization

LOG_DIR = f"{int(time.time())}"

def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

train_dir = "/home/jetson/Desktop/america/archive/seg_train/seg_train/"
val_dir ="/home/jetson/Desktop/america/archive/seg_val/seg_test/"



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  image_size=(32, 32),
  batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  image_size=(32, 32),
  batch_size=32)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def build_model(hp):
        input_shape=(32, 32, 3)
        X_input = Input(input_shape)

        X = ZeroPadding2D((1, 1))(X_input)

        X = Conv2D(hp.Choice('s1_filter_size_1', values=[32,64,128,256],default=32), (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        X = convolutional_block(X, f=3, filters=[hp.Choice('s2_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s2_filter_size_2', values=[32,64,128,256],default=32)], stage=2, block='a', s=1)
        X = identity_block(X, 3, [hp.Choice('s2_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s2_filter_size_2', values=[32,64,128,256],default=32)], stage=2, block='b')
        
        X = convolutional_block(X, f=3, filters=[hp.Choice('s3_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s3_filter_size_2', values=[32,64,128,256],default=32)], stage=3, block='a', s=2)
        X = identity_block(X, 3, [hp.Choice('s3_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s3_filter_size_2', values=[32,64,128,256],default=32)], stage=3, block='b')
       
        X = convolutional_block(X, f=3, filters=[hp.Choice('s4_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s4_filter_size_2', values=[32,64,128,256],default=32)], stage=4, block='a', s=2)
        X = identity_block(X, 3, [hp.Choice('s4_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s4_filter_size_2', values=[32,64,128,256],default=32)], stage=4, block='b')
        
        X = convolutional_block(X, f=3, filters=[hp.Choice('s5_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s5_filter_size_2', values=[32,64,128,256],default=32)], stage=5, block='a', s=2)
        X = identity_block(X, 3, [hp.Choice('s5_filter_size_1', values=[32,64,128,256],default=32), hp.Choice('s5_filter_size_2', values=[32,64,128,256],default=32)], stage=5, block='b')
       
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X= Dropout(hp.Choice('dp_size_1', values=[0.0,0.1,0.2,0.3],default=0.0))(X)
        X = Flatten()(X)
        X=Dense(hp.Choice('dense_size_1', values=[64,128,256],default=64), activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0)), name='fc1',kernel_initializer=glorot_uniform(seed=0))(X)
        X=Dense(hp.Choice('dense_size_2', values=[64,128,256],default=64), activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_2', values=[0.0,0.1, 0.001],default=0.0)), name='fc2',kernel_initializer=glorot_uniform(seed=0))(X)
        X= Dropout(hp.Choice('dp_size_2', values=[0.0,0.1,0.2,0.3],default=0.0))(X)
        X = Dense(6,activation='softmax', name='fc3',kernel_initializer=glorot_uniform(seed=0))(X)

        model = Model(inputs=X_input, outputs=X, name='ResNet18')

        my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=3)]
        opt = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3],default=1e-3))

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                        optimizer=opt,
                                        metrics=['accuracy'])

        return model
        
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=44,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR,
    project_name='BS_ResNet18_Intel')

tuner.search_space_summary()
my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=3)]
tuner.search(train_ds,
             verbose=1, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=10,
             batch_size=64,
             callbacks=[my_callbacks],  # if you have callbacks like tensorboard, they go here.
             validation_data=(val_ds))
             
tuner.results_summary()
with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)
