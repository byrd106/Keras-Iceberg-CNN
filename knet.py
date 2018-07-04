import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout

train_df = pd.read_json("data/processed/train.json")
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

#print x_band1[0].shape

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])

model = Sequential()
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()


model.fit(X_train, y_train, validation_split=0.2)


# model = Sequential()
# model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
# model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.3))
# model.add(Dense(1, activation="sigmoid"))
# model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# model.summary()
