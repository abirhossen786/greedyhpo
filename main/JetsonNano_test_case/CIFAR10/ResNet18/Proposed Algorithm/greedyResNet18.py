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

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images[:10000]
train_labels = train_labels [:10000]
test_images = test_images [:1]
test_labels = test_labels[:1]
train_images = train_images / 255.0
x_train, x_valid, y_train, y_valid = train_test_split(train_images,train_labels,test_size = 0.2, shuffle = True)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
start = timer()
dense_sizes_1 = [64,128,256]
dense_sizes_2 = [64,128,256]
learning_rates_1 = [1e-3,1e-1, 1e-2]
s1_filter_sizes_1 = [32,64,128,256]
s2_filter_sizes_1 = [32,64,128,256]
s2_filter_sizes_2 = [32,64,128,256]
s3_filter_sizes_1 = [32,64,128,256]
s3_filter_sizes_2 = [32,64,128,256]
s4_filter_sizes_1 = [32,64,128,256]
s4_filter_sizes_2 = [32,64,128,256]
s5_filter_sizes_1 = [32,64,128,256]
s5_filter_sizes_2 = [32,64,128,256]
dp_sizes_1 = [0.0,0.1,0.2,0.3]
dp_sizes_2 = [0.0,0.1,0.2,0.3]
wd_sizes_1=[0.0,0.1, 0.001]
wd_sizes_2=[0.0,0.1, 0.001]

opt_layer_1=0.001
opt_layer_2=64
opt_layer_3=64
opt_layer_4=32
opt_layer_5=32
opt_layer_6=32
opt_layer_8=32
opt_layer_9=32
opt_layer_11=32
opt_layer_12=32
opt_layer_14=32
opt_layer_15=32
opt_layer_17=0.0
opt_layer_18=0.0
opt_layer_19=0.0
opt_layer_20=0.0

temp=0
test_acc=0
combinations=0
flag=0
for wd_size_2 in wd_sizes_2 :
	for wd_size_1 in wd_sizes_1 :
		for dp_size_2 in dp_sizes_2 :
			for dp_size_1 in dp_sizes_1 :
				for s5_filter_size_2 in s5_filter_sizes_2:
					for s5_filter_size_1 in s5_filter_sizes_1:
						for s4_filter_size_2 in s4_filter_sizes_2:
							for s4_filter_size_1 in s4_filter_sizes_1 :
								for s3_filter_size_2 in s3_filter_sizes_2:
									for s3_filter_size_1 in s3_filter_sizes_1:
										for s2_filter_size_2 in s2_filter_sizes_2:
											for s2_filter_size_1 in s2_filter_sizes_1:
												for s1_filter_size_1 in s1_filter_sizes_1:
													for dense_size_2 in dense_sizes_2:
														for dense_size_1 in dense_sizes_1:
															for learning_rate_1 in learning_rates_1:
																print(learning_rate_1,dense_size_1,dense_size_2,s1_filter_size_1,s2_filter_size_1,s2_filter_size_2,s3_filter_size_1,s3_filter_size_2,s4_filter_size_1,s4_filter_size_2,s5_filter_size_1,s5_filter_size_2,dp_size_1,dp_size_2,wd_size_1,wd_size_2)
																input_shape=(32, 32, 3)
																X_input = Input(input_shape)

																X = ZeroPadding2D((1, 1))(X_input)

																X = Conv2D(s1_filter_size_1, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
																X = BatchNormalization(axis=3, name='bn_conv1')(X)
																X = Activation('relu')(X)
																X = MaxPooling2D((3, 3), strides=(2, 2))(X)

																X = convolutional_block(X, f=3, filters=[s2_filter_size_1, s2_filter_size_2], stage=2, block='a', s=1)
																X = identity_block(X, 3, [s2_filter_size_1, s2_filter_size_2], stage=2, block='b')


																X = convolutional_block(X, f=3, filters=[s3_filter_size_1, s3_filter_size_2], stage=3, block='a', s=2)
																X = identity_block(X, 3, [s3_filter_size_1, s3_filter_size_2], stage=3, block='b')


																X = convolutional_block(X, f=3, filters=[s4_filter_size_1, s4_filter_size_2], stage=4, block='a', s=2)
																X = identity_block(X, 3, [s4_filter_size_1, s4_filter_size_2], stage=4, block='b')


																X = convolutional_block(X, f=3, filters=[s5_filter_size_1, s5_filter_size_2], stage=5, block='a', s=2)
																X = identity_block(X, 3, [s5_filter_size_1, s5_filter_size_2], stage=5, block='b')

																X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

																X= Dropout(dp_size_1)(X)
																X = Flatten()(X)
																X=Dense(dense_size_1, activation='relu',kernel_regularizer=regularizers.l2(wd_size_1), name='fc1',kernel_initializer=glorot_uniform(seed=0))(X)
																X=Dense(dense_size_2, activation='relu',kernel_regularizer=regularizers.l2(wd_size_2), name='fc2',kernel_initializer=glorot_uniform(seed=0))(X)
																X= Dropout(dp_size_2)(X)
																X = Dense(10,activation='softmax', name='fc3',kernel_initializer=glorot_uniform(seed=0))(X)

																model = Model(inputs=X_input, outputs=X, name='ResNet18')

																my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=3)]
																opt = Adam(learning_rate=learning_rate_1)

																model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
																								optimizer=opt,
																								metrics=['accuracy'])
																model.fit(x_train,y_train,
																		  batch_size=64,
																		  epochs=10,
																		  validation_data=(x_valid,y_valid),
																		  callbacks=[my_callbacks])


																_,test_acc = model.evaluate(x_valid,y_valid)
																print("Current Acc: ",test_acc)
																print("Best Acc: ",temp)
																combinations=combinations+1
																if flag==0 :
																	if test_acc > temp :
																		opt_layer_1 = learning_rate_1
																		temp=test_acc
																	learning_rates_1 = [opt_layer_1]

																elif flag==1 :
																	if test_acc > temp :
																		opt_layer_2 = dense_size_1
																		temp=test_acc
																	dense_sizes_1 = [opt_layer_2]

																elif flag==2 :
																	if test_acc > temp :
																		opt_layer_3 = dense_size_2
																		temp=test_acc
																	dense_sizes_2 = [opt_layer_3]

																elif flag==3 :
																	if test_acc > temp :
																		opt_layer_4 = s1_filter_size_1
																		temp=test_acc
																	s1_filter_sizes_1 = [opt_layer_4]

																elif flag==4 :
																	if test_acc > temp :
																		opt_layer_5 = s2_filter_size_1
																		temp=test_acc
																	s2_filter_sizes_1 = [opt_layer_5]

																elif flag==5 :
																	if test_acc > temp :
																		opt_layer_6 = s2_filter_size_2
																		temp=test_acc
																	s2_filter_sizes_2 = [opt_layer_6]

																elif flag==6 :
																	if test_acc > temp :
																		opt_layer_8 = s3_filter_size_1
																		temp=test_acc
																	s3_filter_sizes_1 = [opt_layer_8]

																elif flag==7 :
																	if test_acc > temp :
																		opt_layer_9 = s3_filter_size_2
																		temp=test_acc
																	s3_filter_sizes_2 = [opt_layer_9]

																elif flag==8 :
																	if test_acc > temp :
																		opt_layer_11 = s4_filter_size_1
																		temp=test_acc
																	s4_filter_sizes_1 = [opt_layer_11]

																elif flag==9 :
																	if test_acc > temp :
																		opt_layer_12 = s4_filter_size_2
																		temp=test_acc
																	s4_filter_sizes_2 = [opt_layer_9]

																elif flag==10 :
																	if test_acc > temp :
																		opt_layer_14 = s5_filter_size_1
																		temp=test_acc
																	s5_filter_sizes_1 = [opt_layer_14]

																elif flag==11 :
																	if test_acc > temp :
																		opt_layer_15 = s5_filter_size_2
																		temp=test_acc
																	s5_filter_sizes_2 = [opt_layer_15]

																elif flag==12 :
																	if test_acc > temp :
																		opt_layer_17 = dp_size_1
																		temp=test_acc
																	dp_sizes_1 = [opt_layer_17]

																elif flag==13 :
																	if test_acc > temp :
																		opt_layer_18 = dp_size_2
																		temp=test_acc
																	dp_sizes_2 = [opt_layer_18]

																elif flag==14 :
																	if test_acc > temp :
																		opt_layer_19 = wd_size_1
																		temp=test_acc
																	wd_sizes_1 = [opt_layer_19]

																elif flag==15 :
																	if test_acc > temp :
																		opt_layer_20 = wd_size_2
																		temp=test_acc
																	wd_sizes_2 = [opt_layer_20]
															flag=1
															test_acc=0
														flag=2
														test_acc=0
													flag=3
													test_acc=0
												flag=4
												test_acc=0
											flag=5
											test_acc=0
										flag=6
										test_acc=0
									flag=7
									test_acc=0
								flag=8
								test_acc=0
							flag=9
							test_acc=0
						flag=10
						test_acc=0  
					flag=11
					test_acc=0
				flag=12
				test_acc=0
			flag=13
			test_acc=0
		flag=14
		test_acc=0
	flag=15
	test_acc=0

end = timer ()
print("######################################################################################################")
print("Trials: ",combinations)
print("Optimal Validation Accuracy",temp)
print("Total Time Elapsed: ", timedelta(seconds=end-start))
print("The Final Optimal Values for all block is : ")
print("learning rate: ",opt_layer_1,"\ndense_size_1: ",opt_layer_2,"\ndense_size_2: ",opt_layer_3,"\ns1_filter_size_1: ",opt_layer_4,"\ns2_filter_size_1: ",opt_layer_5,"\ns2_filter_size_2: ",opt_layer_6,"\ns3_filter_size_1: ",opt_layer_8,"\ns3_filter_size_2: ",opt_layer_9,"\ns4_filter_size_1: ",opt_layer_11,"\ns4_filter_size_2: ",opt_layer_12,"\ns5_filter_size_1: ",opt_layer_14,"\ns5_filter_size_2: ",opt_layer_15,"\ndp_size_1: ",opt_layer_17,"\ndp_size_2: ",opt_layer_18,"\nwd_size_1: ",opt_layer_19,"\nwd_size_2: ",opt_layer_20)
