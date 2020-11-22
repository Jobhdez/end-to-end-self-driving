import scipy
from itertools import islice
from os import listdir
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
import cv2

def train_and_save_model(train_x, train_y, test_x, test_y):

    print("\nMaking Model\n")
    model = make_model()
    
    print("\nTraining Model\n")
    model.fit(train_x, train_y, epochs=100, batch_size=64,
              validation_data=(test_x, test_y), verbose=0)

    print("saving model")
    model.save('autodrive.h5')


def make_model(in_shape=[256, 455, 3], out_shape=1):

    model = VGG16(include_top=False, input_shape=in_shape)

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(out_shape, activation='relu')(class1)

    model = Model(inputs=model.inputs, outputs=output)

    #opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model

def get_datasets(images, angles):
    """given a numpy array of images and a numpy array of angles make
training and testing dataset"""
    split = .8
    length_of_img_list = 1001
    split_index = int(split * length_of_img_list)
    
    train_x = images[:split_index]
    train_y = angles[:split_index]
    test_x = images[split_index:]
    test_y = angles[split_index:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def make_image_list(folder, file_name):
    """given a string denoting a  folder and a string denoting a file_name make a list
of image paths(eg "folder/path.jpg")"""
    images = []
    with open(file_name) as fp:
        for line in islice(fp, None):
            path = line.strip().split()[0]
            image_path = os.path.join(folder, path)
            images.append(image_path)
    
    return images
print(len(make_image_list(folder, file_name)))
def make_angles_numpy(folder, file_name):
    """given a string denoted as folder and a string denoted as file_name make a numpy
array of list of angles"""
    angles = []
    with open(file_name) as fp:
        for line in islice(fp, None):
            angle = line.strip().split()[1].split(",")[0]
            angles.append(float(angle)*scipy.pi/180)
            
    return np.array(angles)

def preprocess_images(list_of_images):
    """given a list consisting of image paths return a numpy array of preprocessed images"""
    def preprocess(img):
        image = cv2.imread(img).astype(np.uint8)[:,:,::-1]
        image = image.astype(np.float32) / 255

        return image

    return np.array([preprocess(img) for _, img in enumerate(list_of_images)])

folder = '/home/square93/Downloads/driving_dataset/driving_dataset2'
file_name = os.path.join(folder, 'data2.txt')
                    
# here, get_datasets consumes two functions: "preprocess_images" and "make_angle_arrays"; the former
# retuns a numpy array of a list of preprocessed images and the latter returns and a numpy array of angles
train_x, train_y, test_x, test_y = get_datasets(preprocess_images(make_image_list(folder, file_name)),
                                                make_angles_numpy(folder, file_name))
# run the model and save it
train_and_save_model(train_x,
                     train_y,
                     test_x,
                     test_y)
    


    

        
        


    
