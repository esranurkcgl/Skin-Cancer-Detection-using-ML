
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

classes = {
    0: ("Melanocytic nevi"),
    1: ("Melanoma"),
    2: ("Benign keratosis-like lesions"),
    3: ("Basal cell carcinoma"),
    4: ("Actinic keratoses"),
    5: ("Vascular lesions"),
    6: ("Dermatofibroma"),
}

import tensorflow as tf
from keras.optimizers import Adam
#resnet152 - 564 layers
SIZE = 100
pre_trained_model = tf.keras.applications.ResNet152V2(include_top=False,
                             input_shape=(SIZE, SIZE, 3),
                             weights='imagenet')

for layer in pre_trained_model.layers[:1]:
    layer.trainable = False
for layer in pre_trained_model.layers[1:]:
    layer.trainable = True


model = tf.keras.models.Sequential([
    pre_trained_model,
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.summary()
model.load_weights("ResNet_model.h5")