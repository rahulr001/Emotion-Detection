import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

data_dir = r'ImgsForTraining\train'

classess = ['0']
# classess = [ 'happy' ]
training_data1 = []
for dir in classess:
    path = os.path.join(data_dir, dir)
    class_index = classess.index(dir)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (224, 224))
            training_data1.append([new_array, class_index])
        except Exception as e:
            pass

random.shuffle(training_data1)
x1 = []
y1 = []
for feature, label in training_data1:
    x1.append(feature)
    y1.append(label)
x1 = np.array(x1).reshape(-1, 224, 224, 3)
x1 = x1 / 255.0
y1 = np.array(y1)

models = tf.keras.applications.MobileNetV3Large()
# new_model = tf.keras.models.load_model('new_trained_model.h5')

base_input = models.layers[0].input
base_output = models.layers[-1].output

final_output_dense = layers.Dense(128)(base_output)
final_output_activate = layers.Activation('relu')(final_output_dense)
final_output_dense = layers.Dense(64)(base_output)
final_output_activate = layers.Activation('relu')(final_output_dense)
final_output_dense = layers.Dense(7, activation= 'softmax')(base_output)

new_model = keras.Model(inputs=base_input, outputs=final_output_dense)

new_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
new_model.fit(x1,y1, epochs=10)

# classess = [ 'fearful', 'happy' ]
# training_data2 = []
# for dir in classess:
#     path = os.path.join(data_dir, dir)
#     class_index = classess.index(dir)
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path, img))
#             new_array = cv2.resize(img_array, (224, 224))
#             training_data2.append([new_array, class_index])
#         except Exception as e:
#             pass

# # temp = np.array(training_data)

# random.shuffle(training_data2)
# x2 = []
# y2 = []
# for feature, label in training_data2:
#     x2.append(feature)
#     y2.append(label)
# x2 = np.array(x2).reshape(-1, 224, 224, 3)
# x2 = x2 / 255.0
# y2 = np.array(y2)
# new_model.fit(x2,y2, epochs=10)


# classess = [  'neutral', 'sad' ]
# training_data3 = []
# for dir in classess:
#     path = os.path.join(data_dir, dir)
#     class_index = classess.index(dir)
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path, img))
#             new_array = cv2.resize(img_array, (224, 224))
#             training_data3.append([new_array, class_index])
#         except Exception as e:
#             pass

# # temp = np.array(training_data)

# random.shuffle(training_data3)
# x3 = []
# y3 = []
# for feature, label in training_data3:
#     x3.append(feature)
#     y3.append(label)
# x3 = np.array(x3).reshape(-1, 224, 224, 3)
# x3 = x3 / 255.0
# y3 = np.array(y3)
# new_model.fit(x3,y3, epochs=10)


# classess = [   'surprised']
# training_data4 = []
# for dir in classess:
#     path = os.path.join(data_dir, dir)
#     class_index = classess.index(dir)
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path, img))
#             new_array = cv2.resize(img_array, (224, 224))
#             training_data4.append([new_array, class_index])
#         except Exception as e:
#             pass

# # temp = np.array(training_data)

# random.shuffle(training_data4)
# x4 = []
# y4 = []
# for feature, label in training_data4:
#     x4.append(feature)
#     y4.append(label)
# x4 = np.array(x4).reshape(-1, 224, 224, 3)
# x4 = x4 / 255.0
# y4 = np.array(y4)

# new_model.fit(x4,y4, epochs=10)
new_model.save('new_trained_model1.h5')