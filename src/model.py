#!/usr/bin/env python

from timeit import default_timer as timer

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.utils import shuffle


np.random.seed(1979)

import tensorflow as tf

layers = tf.keras.layers

from preprocess import batch_generator



def split_train_test(driving_log):
   """
   Split the driving log file to train and validate
   """
   data = pd.read_csv(driving_log,sep='\t')
   data = shuffle(data,random_state=1979)
   df_train, df_valid = model_selection.train_test_split(data,
                                                         test_size=.2,
                                                         random_state=1979)
   return (df_train, df_valid)


pather = "/home/jaganadhg/AI_RND/TF20_eval/driving_dataset/"
data = split_train_test(pather + "data.txt")


def fit_model(data,model_name):
    df_train, df_valid = data

    # use keras to implement NVIDIA architecture,  5 CNN layers, dropout and 4 dense layer

    model = tf.keras.Sequential()
    model.add(layers.Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
    model.add(layers.Cropping2D(cropping=((50,20),(0,0)))) # (top,bottom),(left,right)
    model.add(layers.Conv2D(24,(5,5),strides = (2,2), activation = "relu"))
    model.add(layers.Conv2D(36,(5,5),strides = (2,2), activation = "relu"))
    model.add(layers.Conv2D(48,(5,5),activation = "relu"))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.Dense(50))
    model.add(layers.Dense(10))
    model.add(layers.Dense(1))
    model.compile(loss="mse", optimizer = "adam")
    
    model_start = timer()
    
    mhist = model.fit_generator(
        batch_generator(pather,
                        df_train["center"],
                        df_train["steering"],
                        12),
        steps_per_epoch=10,epochs=100,
        validation_data=batch_generator(pather,
                                        df_valid["center"],
                                        df_valid["steering"],
                                        64),
        validation_steps=int(df_valid.shape[0]/128),
        verbose=1,
        use_multiprocessing=True)
    
    model_end = timer()
    
    print("Time taken for model training is {0}".format(
    model_end - model_start
    ))

    print(mhist.history)

if __name__ == "__main__":
    pather = "/home/jaganadhg/AI_RND/TF20_eval/driving_dataset/"
    data = split_train_test(pather + "data.txt")
    
    
    
    fit_model(data,"Lap6_ep4_model")
    
    
