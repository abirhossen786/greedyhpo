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

from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization
#from kerastuner.engine.hyperparameters import HyperParameters
import pickle

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images[:10000]
train_labels = train_labels [:10000]
test_images = test_images [:1]
test_labels = test_labels[:1]
train_images = train_images / 255.0

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
    model.add(Dense(units=10, activation="softmax"))
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
    max_trials=11,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR,
    project_name='RS_Vgg16_Cifar10')

tuner.search_space_summary()

tuner.search(x=train_images,
             y=train_labels,
             verbose=1, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=10,
             batch_size=64,
             callbacks=[my_callbacks],  # if you have callbacks like tensorboard, they go here.
             validation_split=0.2)

tuner.results_summary()

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


