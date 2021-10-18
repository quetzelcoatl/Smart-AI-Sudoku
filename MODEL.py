import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow._api.v2.v2 import keras

np.random.seed(20)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

datagen = ImageDataGenerator(rotation_range=8,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.10,
                             shear_range=0.15)

train_raw = loadmat('train_32x32.mat')
test_raw = loadmat('test_32x32.mat')

train_images = np.array(train_raw['X'])
test_images = np.array(test_raw['X'])

train_labels = train_raw['y']
test_labels = test_raw['y']

print(train_images.shape)
print(test_images.shape)

train_images = np.moveaxis(train_images, -1, 0)
test_images = np.moveaxis(test_images, -1, 0)

train_images = train_images.astype('float64')
test_images = test_images.astype('float64')
train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')

train_images /= 255.0
test_images /= 255.0

lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                  test_size=0.15, random_state=22)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same',
                        activation='relu',
                        input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation='softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=9.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'second_best_cnn.h5',
    save_best_only=True)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(datagen.flow(X_train, y_train, batch_size=1024),
                    epochs=40, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])
model.save('newmodel2/modelnumber')
