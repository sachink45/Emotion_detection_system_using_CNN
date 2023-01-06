import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import cv2
import sys
sys.path.append('E:\tools\cudnn-windows-x86_64-8.3.1.22_cuda11.5-archive\bin')
from keras.preprocessing.image import ImageDataGenerator

# 
train_data_generator = ImageDataGenerator(rescale= 1./255)
test_data_generator = ImageDataGenerator(rescale= 1./255)

# pre-processing of train_image ->
train_preprocessing = train_data_generator.flow_from_directory(
    'train',
    target_size=(48,48),
    batch_size=(64),
    color_mode='grayscale',
    class_mode='categorical'
)

# pre-processing of test_image ->
test_preprocessing = test_data_generator.flow_from_directory(
    'test',
    target_size=(48,48),
    batch_size=(64),
    color_mode='grayscale',
    class_mode='categorical'
)

# creating cnn model ->

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(7, activation='softmax'))


# compling the cnn model
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.0001, decay = 1e-6), metrics=['accuracy'])

# training the image data with model
model_training = model.fit(train_preprocessing,
                    steps_per_epoch= 28709//64,
                    epochs= 50,
                    validation_data= test_preprocessing,
                    validation_steps= 7178//64)



# save the model structure in json filr.

model_json = model.to_json()
with open('model_json.json', 'w') as json_file:
    json_file.write(model_json)


# save the model weights in h5

model.save_weights('emotion_rec_model.h5')


