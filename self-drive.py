import scipy
from itertools import islice
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import cv2

def train_and_save_model(train_x, train_y, test_x, test_y):

    print("\nMaking Model\n")
    model = make_model()
    
    print("\nTraining Model\n")
    model.fit(train_x, train_y, epochs=100, batch_size=16,
              validation_data=(test_x, test_y), verbose=0)

    print("\nSaving Model\n")
    model.save('autodrive3-1.h5')


def make_model(in_shape=[256, 455, 3], out_shape=1):

    model = VGG16(include_top=False, input_shape=in_shape)

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    FC1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat1)
    FC2 = Dense(64, activation='relu')(FC1)
    output = Dense(out_shape)(FC2)

    model = Model(inputs=model.inputs, outputs=output)

    #opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model

split, data_size = .8, 5001
def get_data(images, angles, split, data_size):
    """ "get_data" consumes a numpy array of images, numpy array of steering angles and returns numpy arrays:
train_x, train_y, test_x, and test_y respectively """

    def get_index(split, data_size): return int(split * data_size)

    return np.array(images[:get_index(split, data_size)]), np.array(angles[:get_index(split, data_size)]), np.array(images[get_index(split, data_size):]), np.array(angles[get_index(split, data_size):])    

def make_list(folder, file_name, function):
    """ consumes a string denoted as "folder", a string denoted as "file_name" and a function;
the function is applied to each line of the textfile"""

    return [function(line) for line in islice(open(file_name), None)]

def make_image_lst(folder, file_name):
    """ uses "make_list" to make a list of image paths"""

    def operate_on_image(line):
        """ given a line of text it extracts the image-name and then joins the string "folder" with the image-name"""
        path = line.strip().split()[0] 
        image_path = os.path.join(folder, path)
        
        return image_path

    return make_list(folder, file_name, operate_on_image)

def make_numpy_of_angles(folder, file_name):
    """ given a folder(string) and a file_name(string) it makes a numpy array of angles"""

    def operate_on_angle(line):
        """ given a line of text it extracts one column(ie the angle) and assigns it to "angle" """
        angle = line.strip().split()[1].split(",")[0]
        angle = float(angle) * scipy.pi/180

        return angle

    return np.array(make_list(folder, file_name, operate_on_angle))
        

def preprocess_images(list_of_images):
    """given a list consisting of image paths return a numpy array of preprocessed images"""
    def preprocess(img):
        image = cv2.imread(img).astype(np.uint8)[:,:,::-1]
        image = image.astype(np.float32) / 255

        return image

    return np.array([preprocess(img) for _, img in enumerate(list_of_images)])

folder = '/home/square93/Downloads/driving_dataset/driving5000'
file_name = os.path.join(folder, 'data.txt')
                    
# here, get_datasets consumes two functions: "preprocess_images" and "make_angle_arrays"; the former
# retuns a numpy array of a list of preprocessed images and the latter returns and a numpy array of angles

train_x, train_y, test_x, test_y = get_data(preprocess_images (make_image_lst(folder, file_name)),
                                            make_numpy_of_angles(folder, file_name),
                                            split,  
                                            data_size) 
                                            
# run the model and save it
train_and_save_model(train_x,
                     train_y,
                     test_x,
                     test_y)


    

        
        


    
