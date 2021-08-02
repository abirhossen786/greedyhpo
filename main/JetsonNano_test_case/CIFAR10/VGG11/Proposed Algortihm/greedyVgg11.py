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
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
#from keras.preprocessing.image import ImageDataGenerator
#import visualkeras
from tensorflow.keras.optimizers import Adam

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images[:10000]
train_labels = train_labels [:10000]
test_images = test_images [:1]
test_labels = test_labels[:1]
train_images = train_images / 255.0
x_train, x_valid, y_train, y_valid = train_test_split(train_images,train_labels,test_size = 0.2, shuffle = True)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
start = timer()
layer_sizes_1 = [32,64,128,256]
layer_sizes_2 = [32,64,128,256]
layer_sizes_3 = [32,64,128,256]
layer_sizes_4 = [32,64,128,256]
layer_sizes_5 = [32,64,128,256]
dense_sizes_1 = [64,128,256]
dense_sizes_2 = [64,128,256]
learning_rates_1 = [1e-3,1e-1,1e-2,]
dp_sizes_6 = [0.0,0.1,0.2,0.3]
wd_sizes_1=[0,0.1, 0.001]
temp=0
test_acc=0
opt_layer_1=32
opt_layer_2=32
opt_layer_3=32
opt_layer_4=32
opt_layer_5=32
opt_layer_6=0.001
opt_layer_7=64
opt_layer_8=64
###############
opt_layer_9=0.0
opt_layer_15=0.0
###############
combinations=0
flag=0
count = False
for wd_size_1 in wd_sizes_1:
	for layer_size_5 in layer_sizes_5:
		for layer_size_4 in layer_sizes_4 :
			for layer_size_3 in layer_sizes_3:
				for layer_size_2 in layer_sizes_2:
					for layer_size_1 in layer_sizes_1:
						for dp_size_6 in dp_sizes_6:
							for dense_size_2 in dense_sizes_2:
								for dense_size_1 in dense_sizes_1:
									for learning_rate_1 in learning_rates_1:
										print(learning_rate_1,dense_size_1,dense_size_2,dp_size_6,layer_size_1,layer_size_2,layer_size_3,layer_size_4,layer_size_5,wd_size_1)
										#####################################Block 1############################################################
										model = Sequential()
										model.add(Conv2D(layer_size_1, (3, 3), activation='relu', padding = 'same', input_shape=(32, 32, 3),kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
										#####################################Block 2######################################################          
										model.add(Conv2D(layer_size_2, (3, 3),padding ='same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
										######################################Block 3##########################################################            
										model.add(Conv2D(layer_size_3, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(Conv2D(layer_size_3, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
										######################################Block 4######################################################## 
										model.add(Conv2D(layer_size_4, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(Conv2D(layer_size_4, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
										######################################Block 5######################################################## 
										model.add(Conv2D(layer_size_5, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(Conv2D(layer_size_5, (3, 3), padding = 'same', activation='relu',kernel_regularizer=regularizers.l2(wd_size_1)))  
										model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
										######################################################################################################
										model.add(Dropout(dp_size_6))
										model.add(Flatten())
										model.add(Dense(units=dense_size_1,activation="relu",kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(Dense(units=dense_size_2,activation="relu",kernel_regularizer=regularizers.l2(wd_size_1)))
										model.add(Dense(units=10, activation="softmax"))
										#######################################################################################################            

										my_callbacks = [ tf.keras.callbacks.EarlyStopping(patience=3)]
										opt = Adam(learning_rate=learning_rate_1)

										model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
													  optimizer=opt,
													  metrics=['accuracy'])

										hist=model.fit(x_train,y_train,
												  batch_size=64,
												  epochs=10,
												  validation_data=(x_valid,y_valid),
												  callbacks=[my_callbacks])

										_,test_acc = model.evaluate(x_valid, y_valid)

										print("Current Acc: ",test_acc)
										print("Best Acc: ",temp)
										combinations=combinations+1

										if flag==0 :
											if test_acc > temp :
												opt_layer_6 = learning_rate_1
												temp=test_acc
											learning_rates_1 = [opt_layer_6]

										elif flag==1 :
											if test_acc > temp :
												opt_layer_7 = dense_size_1
												temp=test_acc 
											dense_sizes_1 = [opt_layer_7]

										elif flag==2 :
											if test_acc > temp :
												opt_layer_8 = dense_size_2
												temp=test_acc
											dense_sizes_2 = [opt_layer_8]
										###############################################################################
										elif flag==3 :
											if test_acc > temp :
												opt_layer_9 = dp_size_6
												temp=test_acc
											dp_sizes_6 = [opt_layer_9]

										elif flag==4 :
											if test_acc > temp :
												opt_layer_1 = layer_size_1
												temp=test_acc
											layer_sizes_1 = [opt_layer_1]
										###############################################################################
										elif flag==5:
											if test_acc > temp :
												opt_layer_2 = layer_size_2
												temp=test_acc
											layer_sizes_2 = [opt_layer_2]

										elif flag==6:
											if test_acc > temp :
												opt_layer_3 = layer_size_3
												temp=test_acc
											layer_sizes_3 = [opt_layer_3]

										elif flag==7:
											if test_acc > temp :
												opt_layer_4  = layer_size_4
												temp=test_acc
											layer_sizes_4 = [opt_layer_4]

										elif flag==8:
											if test_acc > temp :
												opt_layer_5  = layer_size_5
												temp=test_acc
											layer_sizes_5 = [opt_layer_5]

										elif flag==9:
											if test_acc > temp :
												opt_layer_15 = wd_size_1
												temp=test_acc
											wd_sizes_1 = [opt_layer_15]

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


end = timer ()

print("######################################################################################################")
print("Trials : ",combinations)
print("Optimal Validation Accuracy: ",temp)
print("Total Time Elapsed: ", timedelta(seconds=end-start))
print("The Final Optimal Values for all block is : ")
print('learning_rate_1: ',opt_layer_6,'\ndense_size_1: ',opt_layer_7,'\ndense_size_2: ',opt_layer_8,'\ndp_size_6: ',opt_layer_9,'\nlayer_size_1: ',opt_layer_1,'\nlayer_size_2: ',opt_layer_2,'\nlayer_size_3: ',opt_layer_3,'\nlayer_size_4: ',opt_layer_4,'\nlayer_size_5: ',opt_layer_5,'\nweight_decay: ',opt_layer_15)

