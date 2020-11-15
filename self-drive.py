import scipy
from itertools import islice
from os import listdir
import os
from matplotlib import pyplot
import matplotlib.image as image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
from keras import backend as K
import tensorflow as tf
"""
def make_img_array2(folder):
    images = []
    for img in listdir(folder):
        img_path = os.path.join(folder + "/" + img)
        images.append(img_path)
    return images
"""

def make_img_array(folder):
    images = []
    for img in sorted(listdir(folder),key=lambda x: int(os.path.splitext(x)[0])):
        img = os.path.join(folder + "/" + img)
        img = cv2.imread(img).astype(np.uint8)[:,:,::-1]
        #images = np.array(images)
        #images = np.resize(images, (128, 128, 3))
        img = img.astype(np.float32) / 255
        images.append(img)
        
        
    return np.array(images)

# Angles is a numpy array of angles

# create_angle_array: Folder Training_Data -> Angles
# given a Folder and training data the function returns an array consisting
# of angles
def create_angle_array(Training_Data):
    training_file = open(Training_Data)
    angle_array = []
    for line in training_file:
        angle = line.split()
        angle = angle[1]
        angle = angle.split(",")
        angle = angle[0]
        angle_array.append(float(angle)*scipy.pi/180)
    
    
    return np.array(angle_array)


# Images is an array(ie data images)
# Numpys is an array(the corresponding numpy array of Images)

# img-to-numpy: Images -> Numpys
# given an array of imgs this function returns it corresponding numpy array
"""
def img_to_numpy(Images):
    Numpys = [get_numpys_ready(img) for img in Images]
    Numpys = np.array(Numpys)
    return Numpys

def get_numpys_ready(img):
    images = cv2.imread(img).astype(np.uint8)[:,:,::-1]
    #images = np.array(images)
    #images = np.resize(images, (128, 128, 3))
    images = images.astype(np.float32) / 255
    return images
"""
def get_features(folder):
    features = make_img_array(folder)
    
    return features

def get_labels(training_data):
    labels = create_angle_array(training_data)
    
    return labels


#folder = '/Users/hdez/Downloads/driving_dataset'
#file_name = os.path.join(folder, 'data.txt')

"""
images = []
angles = []
with open(file_name) as fp:
    for line in islice(fp, None):
        path = line.strip().split()[0]
        angle = line.strip().split()[1].split(",")[0]
        image_path = os.path.join(folder + "/" + path)
        images.append(image_path)
        angles.append(float(angle)*scipy.pi/180)

print("processing imgs")
imgs = []
for i in range(len(images)):
    img = cv2.imread(images[i]).astype(np.uint8)[:,:,::-1]
    img = img.astype(np.float32) / 255
    imgs.append(img)
"""

#print("turning imgs to a numpy array")
#imagess = np.array(imgs)
#angles = np.array(angles)

def make_model(in_shape=[256, 455, 3], out_shape=17):

    model = VGG16(include_top=False, input_shape=in_shape)

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(out_shape, activation='sigmoid')(class1)

    model = Model(inputs=model.inputs, outputs=output)

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    return model

def get_datasets(folder, training_data):

    split = .8
    print("assigment of features and labels")
    features = get_features(folder)
    labels = get_labels(training_data)
    print("exiting assignment of features and labels")

    split_index = int(split * 45406)

    train_x = features[:split_index]
    train_y = labels[:split_index]
    test_x = features[split_index:]
    test_y = labels[split_index:]
    

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
                    
def train_and_evaluate_model(folder, training_data):

    train_x, train_y, test_x, test_y = get_datasets(folder, training_data)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    #train_x = np.asarray(train_x).astype('float32') / 255
    #train_y = np.asarray(train_y).astype('float32') / 255
   # print(train_x.shape, train_y.shape)
    #train_x = np.resize(train_x, (73728,73728))
    #train_x = np.resize(train_y, (73728,73728))
    #train_x = np.asarray(train_x).astype('float32')
    #train_y = np.asarray(train_y).astype('float32')                   
    #train_x = np.reshape(train_x, (128, 128, 3))
    #train_y = np.reshape(train_y, (128, 128, 3))

    print("\nMaking Model\n")
    model = make_model()
    print("\nTraining Model\n")

    """what exactly are the types of model.fit()"""
    model.fit(train_x, train_y, epochs=100, batch_size=64,
              validation_data=(test_x, test_y), verbose=0)

    print("saving model")
    model.save('auto.h5')

    
    

folder = '/Users/hdez/Downloads/driving_dataset'
training_data = 'data.txt'

train_and_evaluate_model(folder, training_data)
    
    


    

        
        


    
