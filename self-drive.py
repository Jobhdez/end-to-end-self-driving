import pandas as pd
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Model

def run_model(imgs_directory, file_name):

    training_datagen = ImageDataGenerator(rescale = 1./255,
                                      fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)

    dataframe = create_filename_column(get_dataframe(file_name))

    split = 0.8

    training, validation = get_datasets(dataframe, split)

    train_generator = get_training_datagen(training_datagen, training, images_directory, (256, 455), 32)

    val_generator = get_validation_datagen(validation_datagen, validation, images_directory, (256, 455))

    train_and_save_model(train_generator, val_generator)


def train_and_save_model(train_generator, val_generator):

    print("\nMaking Model\n")
    model = make_model()
    
    print("\nTraining Model\n")
    model.fit(train_generator, verbose = 1, validation_data=val_generator, epochs=100)
    print("\nSaving Model\n")
    model.save('autodrive45k.h5')

def make_model(in_shape=[256, 455, 3], out_shape=1):

    model = VGG16(include_top=False, input_shape=in_shape)

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    FC1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat1)
    FC2 = Dense(64, activation='relu')(FC1)
    output = Dense(out_shape, activation='linear')(FC2)

    model = Model(inputs=model.inputs, outputs=output)
    #opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model

def get_dataframe(file_name):
    """ given a string(csv file) it reads and creates dataframe"""

    return pd.read_csv(file_name,sep=r'\s*,\s*', na_values=['NA', '?'])

def create_filename_column(dataframe):

    """it adds a column named 'filename'; if you were to print this data frame it would have 
3 columns: id, steering_angle, and filename"""

    dataframe['filename'] = dataframe["id"].astype(str)

    return dataframe

def get_datasets(dataframe, split):
    """given a dataframe and a split it returns the training dataset and validation dataset"""

    def get_index():
        return int(len(dataframe) * split)

    return dataframe[0:get_index()], dataframe[get_index():]

def get_training_datagen(generator, training, img_directory, targetsize, batchsize):

    """generator for progressively loading training data"""

    return generator.flow_from_dataframe(dataframe=training,
                                         directory=img_directory,
                                         x_col="filename",
                                         y_col="steering_angle",
                                         target_size=targetsize,
                                         batch_size=batchsize,
                                         class_mode='other')

def get_validation_datagen(generator, validation, img_directory, targetsize):

    """generator for progressively loading validation data"""

    return generator.flow_from_dataframe(dataframe=validation,
                                         directory=img_directory,
                                         x_col="filename",
                                         y_col="steering_angle",
                                         target_size=targetsize,
                                         class_mode='other')


images_directory = "/home/square93/Downloads/driving_dataset/data45406"
file_name = "/home/square93/Downloads/driving_dataset/data.csv"

run_model(images_directory, file_name)
        


    
