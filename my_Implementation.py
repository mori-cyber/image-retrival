import tensorflow as tf
# import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
import caer
# import canaro
import cv2 as cv
conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# tf.keras.utils.plot_model(conv_base, show_shapes=True)
# --------------------------------------------------------------------------------------------------
# conv_base.summary()
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = r'C:\Users\diana\Pictures\Data\dogcat'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 25


# ------------------------------------------------------------------------------------------------------

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop, # we must `b
            # reak` after every
            break
    return features, labels


# -----------------------------------------------------------------------------------------------------------

train_features, train_labels = extract_features(train_dir, 200)
validation_features, validation_labels = extract_features(validation_dir, 100)
test_features, test_labels = extract_features(test_dir, 100)
# ----------------------------------------------------------------------------------------------------------
train_features = np.reshape(train_features, (200, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (100, 4 * 4 * 512))
test_features = np.reshape(test_features, (100, 4 * 4 * 512))
# ----------------------------------------------------------------------------------------------------------
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

# ---------------------------------------------------------------------------------------------------------

history = model.fit(train_features, train_labels,
                    epochs=10,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

evalu_test = model.evaluate(test_features, test_labels)

print('this is evaluation of test', evalu_test)

# --------------------------------------------------------------------------------------------------------



# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
#
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show(block=True)
# plt.interactive(False)
# -------------------------------------------------------------------------------------------------------



# test_path =r'C:\Users\diana\Pictures\Data\dogcat\test\test1\20.jpg'
img = cv.imread(r'C:\Users\diana\Pictures\Data\dogcat\test\test1\20.jpg')
# plt.imshow('a',img)
IMG_SIZE=(150,150)
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# f = cv.imread('C:/Users/Music/university_project/Data/images1.jpg')
    img = cv.resize(img, IMG_SIZE)

    img = img / 255.0
    img = caer.reshape(img, IMG_SIZE,3)
    return img
preduction = model.predict(prepare(img))
print(np.argmax(preduction))
# y_hat = model.predict(img)
# print((y_hat))
cv.imshow(img)
plt.show(img)
