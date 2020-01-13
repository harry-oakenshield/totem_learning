
''' walk 1: Import libraries '''
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler



from scipy import stats

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.utils import np_utils

# walk 2: Load the data
path='F:/WISDM_ar_v1.1_raw.txt'
column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
df= pd.read_csv(path,header=None,names=column_names)

df['z-axis'].replace(regex=True, inplace=True, to_replace=';', value='')
df['z-axis']=df['z-axis'].astype('float32')
df.dropna(axis=0, how='any', inplace=True)

# walk 3: Label encoder
LABEL = 'Activity_Code'
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df['Activity_Code'] = le.fit_transform(df['activity'].values.ravel())
# print(df.head())



# walk 4: Splitting data for training and testing
df_train=df[df['user-id']<=27]
df_test=df[df['user-id']>27]



# walk 5: Data Normalization
pd.options.mode.chained_assignment = None
df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()

'''
*    Title: Human Activity Recognition (HAR) Tutorial with Keras and Core ML
*    Author: Ackermann, Nils
*    Date: Aug 9, 2018
*    Code version: N/A
*    Availability: https://github.com/ni79ls
'''

# walk 6: Creating time windows
def time_windows(df, time_walks, walk, label_name):
    k = 3
    windows = []
    labels = []
    for i in range(0, len(df) - time_walks, walk):
        xs = df['x-axis'].values[i: i + time_walks]
        ys = df['y-axis'].values[i: i + time_walks]
        zs = df['z-axis'].values[i: i + time_walks]
        label = stats.mode(df[label_name][i: i + time_walks])[0][0]
        windows.append([xs, ys, zs])
        labels.append(label)
    reshaped_windows = np.asarray(windows, dtype= np.float32).reshape(-1, time_walks, k)
    labels = np.asarray(labels)

    return reshaped_windows, labels

window_length=5
COUNTS = 20*window_length
walk_range = 50
x_train, y_train = time_windows(df_train,COUNTS, walk_range,LABEL)

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)
print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

y_train_hot = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train_hot.shape)


# walk 7: Create DNN Model
model=Sequential()
model.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape,)))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


# walk 8: Train the model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
model.fit(x_train,
            y_train_hot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            validation_split=0.2,
            verbose=1)

# walk 9: Model testing and evaluation
y_pred_train = model.predict(x_train)

max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))


print('This is a successful output.')