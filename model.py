import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import random

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

samples = shuffle(lines)

# splitting samples into training and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# random brightness augmentation function
def augment_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

# to resize images
TARGET_SIZE = (64, 64)
def resize_to_target_size(image):
    return cv2.resize(image, TARGET_SIZE)

# to return cropped and resized images
def crop_and_resize(image):
    '''
    image: The input image of dim 160x320x3
    return: Output image of dim 64x64x3
    '''
    cropped_image = image[55:135, :, :]
    processed_image = resize_to_target_size(cropped_image)
    return processed_image

# to normalize images
def normalize(image):
    return ((image - 255.0) - 0.5)

# to return 1 randomly augmented image sample from each row of data after preprocessing
def get_augmented_sample(line):
    steering_center = float(line[3])

    # randomly choose either center, left or right images
    choice = random.choice(['center', 'left', 'right'])

    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    path = "data/IMG/"

    if choice == 'left':
        image = cv2.imread(path + line[1].split('/')[-1])
        angle = steering_center + correction
    
    elif choice == 'right':
        angle = steering_center - correction
        image = cv2.imread(path + line[2].split('/')[-1])

    elif choice == 'center':
        image = cv2.imread(path + line[0].split('/')[-1])
        angle = steering_center
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # randomly choosing between normal image or flipped image
    if random.choice(['yes', 'no']) == 'yes':
        image = np.fliplr(image)
        angle = -angle

    # using brightness augmentation
    image = augment_brightness(image)

    # crop and resize
    image = crop_and_resize(image).astype(np.float32)

    # normalize image
    image = normalize(image)

    return image, angle

# generator function to generate images batchwise on the fly for training and validating the model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    batches_per_epoch = num_samples // batch_size
    
    i = 0
    while True: # Loop forever so the generator never terminates
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
        y_batch = np.zeros((batch_size), dtype = np.float32)

        j = 0
        
        # slice a batch_size sized chunk from samples
        for sample in samples[start:end]:
            X_batch[j], y_batch[j] = get_augmented_sample(sample)
            j+=1

        i+=1

        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle again over samples
            i = 0

        yield X_batch, y_batch

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model 
from keras.layers import Flatten, Dense, Conv2D, Cropping2D, MaxPooling2D, ELU
from keras.layers.core import Lambda, Dropout

# model architecture (from NVIDIA's end-to-end deep learning paper)
batch_size = 32

model = Sequential()

model.add(Conv2D(24,5,5,subsample=(1,1),activation='relu',input_shape=(64,64,3)))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,subsample=(1,1),activation='relu'))
model.add(Conv2D(64,3,3,subsample=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=(20000.0//batch_size)*batch_size , \
            validation_data=validation_generator, \
            nb_val_samples=3000, nb_epoch=7)

# saving the model
model.save('model.h5')
