import matplotlib.pyplot as plt
import tensorflow
# from tensorflow import keras
# import keras



import tensorflow.keras as keras
import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------

from tensorflow.keras.applications.resnet50 import ResNet50
conv_base = ResNet50(weights='imagenet',
include_top=False,
input_shape=(32, 32, 3))

# -----------------------------------------------------------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.datasets import cifar10
# base_dir=cifar10.load_data()
base_dir =r'C:\Users\diana\Pictures\Data\dogcat'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 25

# _______________________________________________________________________________________

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 5,5,2048))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary'
        )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
        # Note that since generators yield data indefinitely in a loop, # we must `b
        # reak` after every image has been seen once.
            break
    return features, labels

# ---------------------------------------------------------------------------------------
train_features, train_labels = extract_features(train_dir, 400)
validation_features, validation_labels = extract_features(validation_dir, 100)
test_features, test_labels = extract_features(test_dir, 100)

# --------------------------------------------------------------------------------------

train_features = np.reshape(train_features, (400, 5*5*2048))
train_features = train_features.astype('float32')/255.0-0.5
validation_features = np.reshape(validation_features, (100, 5*5*2048))
validation_features = validation_features.astype('float32')/255.0-0.5
test_features = np.reshape(test_features, (100, 5*5*2048))
test_features = test_features.astype('float32')/255.0-0.5

# ___________________________________________________________________________________________


model = models.Sequential()
model.add(layers.Dense(256, activation='elu', input_dim=5*5*2048))
model.add(layers.Dense(128, activation='elu'))
model.add(layers.Dense(64, activation='elu'))
# model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['acc'])

# ______________________________________________________________________________________

history = model.fit(train_features, train_labels,
epochs=10,
batch_size=20,
validation_data=(validation_features, validation_labels))

# ________________________________________________________________________________________

test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=1)
print('\nTest accuracy:', test_acc)
# print(ndim.test_features[20])

# ___________________________________________________________________________________________
import sklearn
images = train_features
codes = model.predict(images)
assert  len(codes) == len(images)
from sklearn.neighbors._unsupervised import  NearestNeighbors
nei_clf = NearestNeighbors(metric='euclidean')
nei_clf.fit(codes)

# ----------------------------------------------------------------------------------------

def get_similar(image,n_neighbors=5):
    # try:
    # image = plt.imread(image)
    assert image.ndim == 1
    code = model.predict(image[0])
    (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors = n_neighbors)
    # except AssertionError:
    #     print('i amangry')
    #     exit(1)
    return distances,images[idx]
# ----------------------------------------------------------------------------------------
def show_image(x):
    plt.imshow(np.clip(x+0.5,0,1))

def show_similar(image):
    distaces ,neighbors = get_similar(image,n_neighbors = 3)
    plt.figure(figsize=[6,7])
    plt.subplot(1,4,1)
    show_image(image)
    plt.title("Original image")
    for i in range(3):
        plt.subplot(1,4,i+2)
        show_image(neighbors[i])
        plt.title("Dis=%3f"% distaces[i])
    plt.show()

# -------------------------------------------------------------
# print(np.ndim(test_features[50]))
show_similar(test_features[50])


