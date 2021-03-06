import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


train_df = pd.read_json("data/processed/train.json")
#test_df = pd.read_json("data/processed/test.json")

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

#plot_model(model, to_file='ogmodel.png')

#e = 150
e = 1
history = model.fit(X_train, y_train, validation_split=0.2,epochs=e)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.savefig(str(e)+'_ThirdAccgraph.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.savefig(str(e)+'_Thirdlossgraph.png')


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
# submit_df.to_csv("./second_submission.csv", index=False)



