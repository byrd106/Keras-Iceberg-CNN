import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from helpers import *


import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.utils import plot_model
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


train_df = pd.read_json("data/processed/train.json")
test_df = pd.read_json("data/processed/test.json")

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])

# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True) 
# datagen.fit(X_train)
# results = datagen.flow(X_train,y_train,batch_size=2000)
# newImages = []
# for k in results:
# 	newImages = k
# 	break

# X_train = np.append(X_train,newImages[0],axis=0)
# y_train = np.append(y_train,newImages[1])

print X_train.shape 
print y_train.shape


model = Sequential()

model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.2))

model.add(Convolution2D(128, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

model.add(Dense(200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
optimizer = Adam(decay=0.01)
model.summary()

#plot_model(model, to_file='ogmodel.png')

e = int(sys.argv[2])
#e = 1

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(X_train, y_train, validation_split=0.2,epochs=e)
#,callbacks=[tbCallBack])

netname = sys.argv[1]
print netname
savePlot("A",history,netname,e)
savePlot("L",history,netname,e)

# this net gets about a .6558 on the leaderboard (1 EPOCH!!!) 
# now w/ 50 epocs, see score increase 
# next round of testing: 
# Test data

# x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
# x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
# X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
# print("Xtest:", X_test.shape)

# prediction = model.predict(X_test, verbose=1)
# submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
# submit_df.to_csv("./HeyMOREIMAGESsevennet_submission.csv", index=False)
 


