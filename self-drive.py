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

def train_and_save_model():

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

dataframe = pd.read_csv("/home/square93/Downloads/driving_dataset/data2.csv",
                        sep=r'\s*,\s*',
                        na_values=['NA', '?'])

dataframe['filename']= dataframe["id"].astype(str)

split = 0.8
index = int(len(dataframe) * split)

df_train = dataframe[0:index]
df_validate = dataframe[index:]


images_directory = "/home/square93/Downloads/driving_dataset/data10k"

training_datagen = ImageDataGenerator(rescale = 1./255,
                                      fill_mode='nearest')

train_generator = training_datagen.flow_from_dataframe(dataframe=df_train,
                                                       directory=images_directory,
                                                       x_col="filename",
                                                       y_col="steering_angle",
                                                       target_size=(256, 455),
                                                       batch_size=32,
                                                       class_mode='other')
validation_datagen = ImageDataGenerator(rescale=1./255)

val_generator = validation_datagen.flow_from_dataframe(dataframe=df_validate,
                                                       directory=images_directory,
                                                       x_col="filename",
                                                       y_col="steering_angle",
                                                       target_size=(256, 455),
                                                       class_mode='other')



    

        
        


    
