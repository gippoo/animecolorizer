import numpy as np
from PIL import Image
from keras.layers import Conv2D, UpSampling2D, InputLayer, Input, concatenate, Dropout
from keras.models import Model
import matplotlib.pyplot as plt

# Load the images and store as a numpy array
# One set of the original RGB values and one set of the grayscale values
color_imgs = []
gs_imgs = []

for i in range(462):
    img = Image.open('.../anime-faces/blue_hair/{}.jpg'.format(i))
    imgdata = np.asarray(list(img.getdata())).reshape(96, 96, 3)
    color_imgs.append(imgdata)

for i in range(462):
    img = Image.open('.../anime-faces/blue_hair/{}.jpg'.format(i)).convert('L')
    imgdata = np.asarray(list(img.getdata())).reshape(96, 96)
    gs_imgs.append(imgdata)

# Convert the lists to numpy arrays and normalize values to be between 0 and 1
color_imgs = np.asarray(color_imgs)/255
gs_imgs = 1 - np.asarray(gs_imgs)/255


# Sanity check to make sure images loaded properly
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(color_imgs[i])
plt.show()

plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(gs_imgs[i], cmap='Greys')
plt.show()


# Split data to training and validation images
# 412 used for training and the remaining 50 used for validation
gs_train = gs_imgs[:412]
gs_test = gs_imgs[412:]
color_train = color_imgs[:412]
color_test = color_imgs[412:]


# Reshape grayscale matrix to be 4 dimensional so the model accepts it as input.
gs_train = gs_train.reshape(-1, 96, 96, 1)
gs_test = gs_test.reshape(-1, 96, 96, 1)


# Build the model
input_img = Input(shape=(96, 96, 1))
conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv2)
conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv3)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv4)
up7 = concatenate([UpSampling2D((2, 2))(conv4), conv3])
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
conv7 = Dropout(0.2)(conv7)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
conv7 = Dropout(0.2)(conv7)
up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2])
conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1])
conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

model = Model(input_img, conv10)
model.compile(optimizer='RMSprop', loss='mae')


# Train the model
model.fit(gs_train, color_train,
          epochs=60,
          batch_size=8,
          shuffle=True,
          validation_data=(gs_test, color_test))


# Show results on validation data
plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(model.predict(gs_test[i].reshape(1, 96, 96, 1)).reshape(96, 96, 3))
plt.show()
