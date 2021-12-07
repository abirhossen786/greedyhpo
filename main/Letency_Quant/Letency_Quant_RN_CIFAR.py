import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np
import time
import pathlib
import timeit
import seaborn as sns

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# Normalize pixel values to be between 0 and 1
test_images = test_images / 255.0

## No_quant

testcode = ''' 
def test(): 
    model = keras.models.load_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/greedy_approch_Cifar10_ResNet50')
    model.predict(test_images[0])
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("NQ_greedy_approch_Cifar10_ResNet50.csv")
print("Latency saved...")

testcode = ''' 
def test(): 
    model = keras.models.load_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Bayesian_Search_Cifar10_ResNet50')
    model.predict(test_images[0])
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("NQ_Bayesian_Search_Cifar10_ResNet50.csv")
print("Latency saved...")

testcode = ''' 
def test(): 
    model = keras.models.load_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Random_Search_Cifar10_ResNet50')
    model.predict(test_images[0])
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("NQ_Random_Search_Cifar10_ResNet50.csv")
print("Latency saved...")

#Q1

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/greedy_approch_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q1_greedy_approch_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q1_greedy_approch_Cifar10_ResNet50.csv")
print("Latency saved...")

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Bayesian_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q1_Bayesian_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q1_Bayesian_Search_Cifar10_ResNet50.csv")
print("Latency saved...")

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Random_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q1_Random_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q1_Random_Search_Cifar10_ResNet50.csv")
print("Latency saved...")

#Q2

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/greedy_approch_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q2_greedy_approch_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q2_greedy_approch_Cifar10_ResNet50.csv")
print("Latency saved...")

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Bayesian_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q2_Bayesian_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q2_Bayesian_Search_Cifar10_ResNet50.csv")
print("Latency saved...")

converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Random_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q2_Random_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q2_Random_Search_Cifar10_ResNet50.csv")
print("Latency saved...")


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [tf.dtypes.cast(input_value, tf.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/greedy_approch_Cifar10_ResNet50')

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q3_greedy_approch_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q3_greedy_approch_Cifar10_ResNet50.csv")
print("Latency saved...")


converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Bayesian_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q3_Bayesian_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()


testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q3_Bayesian_Search_Cifar10_ResNet50.csv")
print("Latency saved...")


converter = tf.lite.TFLiteConverter.from_saved_model('/home/anjir29/Desktop/greedyhpo-main/Main_test_case/CIFAR10/Random_Search_Cifar10_ResNet50')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

tflite_models_dir = pathlib.Path("Quant_Models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"Q3_Random_Search_Cifar10_ResNet50.tflite"
tflite_model_file.write_bytes(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter_quant.allocate_tensors()

testcode = ''' 
def test(): 
    test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
'''
time = timeit.repeat(stmt=testcode, repeat=100)
#time = np.array(time)
time = np.reshape(time, (100, 1))

#print(time)

pd.DataFrame(time).to_csv("Q3_Random_Search_Cifar10_ResNet50.csv")
print("Latency saved...")


