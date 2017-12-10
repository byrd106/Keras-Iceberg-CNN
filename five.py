import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import train_test_split


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.utils import plot_model

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())




train = pd.read_json("data/processed/train.json")
#test_df = pd.read_json("data/processed/test.json")

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
Y_train=train['is_iceberg']

model=Sequential()
    
# CNN 1
model.add(Conv2D(8, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

# CNN 2
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

# CNN 3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

#CNN 4
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# You must flatten the data for the dense layers
model.add(Flatten())

#Dense 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Dense 2
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# Output 
model.add(Dense(1, activation="sigmoid"))

optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.summary()

#plot_model(model, to_file='ogmodel.png')

#e = 150
e = 200
batch_size = 32

X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, Y_train, random_state=1, train_size=0.75)

history = model.fit(X_train_cv, y_train_cv, batch_size=batch_size, epochs=e, validation_data=(X_valid, y_valid), validation_split=0.2)



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(str(e)+'_fiveAccgraph.png')
plt.clf()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.savefig(str(e)+'_fivelossgraph.png')

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



