import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Input
)

# Build Same CNN Model
model = Sequential()

model.add(Input(shape=(48,48,1)))

# Block 1
model.add(Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.30))

# Fully Connected
model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Dense(7,activation="softmax"))


# Load Weights
model.load_weights("emotion.weights.h5")
print("Model Loaded Successfully")
