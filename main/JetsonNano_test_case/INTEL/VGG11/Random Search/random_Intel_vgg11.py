import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

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

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization

import pickle

from keras.preprocessing.image import ImageDataGenerator

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

LOG_DIR = f"{int(time.time())}"

def build_model(hp):

    model = Sequential()
    model.add(Conv2D(hp.Choice('layer_size_1', values=[32,64,128,256],default=32), 3, activation='relu', padding = 'same', input_shape=(32, 32, 3),kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #####################################Block 2#################################         
    model.add(Conv2D(hp.Choice('layer_size_2', values=[32,64,128,256],default=32), 3,padding ='same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    ######################################Block 3#################################            
    model.add(Conv2D(hp.Choice('layer_size_3', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001 ],default=0.0))))
    model.add(Conv2D(hp.Choice('layer_size_3', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    ######################################Block 4################################
    model.add(Conv2D(hp.Choice('layer_size_4', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(Conv2D(hp.Choice('layer_size_4', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    ######################################Block 5#################################
    model.add(Conv2D(hp.Choice('layer_size_5', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(Conv2D(hp.Choice('layer_size_5', values=[32,64,128,256],default=32), 3, padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    ############################################################################
    model.add(Dropout(hp.Choice('dp_size_6', values=[0.0,0.1,0.2,0.3],default=0.0)))
    model.add(Flatten())
    model.add(Dense(hp.Choice('dense_size_1', values=[64,128,256],default=64),activation="relu",kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(Dense(hp.Choice('dense_size_2', values=[64,128,256],default=64),activation="relu",kernel_regularizer=regularizers.l2(hp.Choice('wd_size_1', values=[0.0,0.1, 0.001],default=0.0))))
    model.add(Dense(units=6, activation="softmax"))
    ##############################################################################           
    
    opt = Adam(lr=hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3],default=1e-3))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                                  optimizer=opt,
                                                  metrics=['accuracy'])
    return model

my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=3)]

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=27,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR,
    project_name='RS_Vgg16_Intel')

tuner.search_space_summary()

tuner.search(train_ds,
             verbose=1, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=10,
             batch_size=64,
             callbacks=[my_callbacks],  # if you have callbacks like tensorboard, they go here.
             validation_data=val_ds)

tuner.results_summary()

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


