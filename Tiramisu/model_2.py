#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tiramisu-keras github
# https://github.com/junjungoal/Tiramisu-keras/blob/master/Tiramisu.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import pydot
import graphviz

def denseBlock(t, nb_layers):
    for _ in range(nb_layers):
        tmp = t
        t = BatchNormalization(axis=1,
                                gamma_regularizer=l2(0.0001),
                                beta_regularizer=l2(0.0001))(t)

        t = Activation('relu')(t)
        t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
        t = Dropout(0.2)(t)
        t = concatenate([t, tmp])
    return t

def transitionDown(t, nb_features):
    t = BatchNormalization(axis=1,
                            gamma_regularizer=l2(0.0001),
                            beta_regularizer=l2(0.0001))(t)
    t = Activation('relu')(t)
    t = Conv2D(nb_features, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
    t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last')(t)
    return t

def my_model(width, height, layer_per_block, n_pool=5, growth_rate=16):
    input = Input(shape=(width, height, 3))
    t = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)

    #dense block
    nb_features = 48
    skip_connections = []
    for i in range(n_pool):
        t = denseBlock(t, layer_per_block[i])
        skip_connections.append(t)
        nb_features += growth_rate * layer_per_block[i]
        t = transitionDown(t, nb_features)

    t = denseBlock(t, layer_per_block[n_pool]) # bottle neck

    skip_connections = skip_connections[::-1] #subvert the array

    for i in range(n_pool):
        keep_nb_features = growth_rate * layer_per_block[n_pool + i]
        t = Conv2DTranspose(keep_nb_features, strides=2, kernel_size=(3, 3), padding='same', data_format='channels_last')(t) # transition Up
        t = concatenate([t, skip_connections[i]])

        t = denseBlock(t, layer_per_block[n_pool+i+1])

    output_layer = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
    
    
    model = Model(inputs=input, outputs=output_layer)
    return model

#layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]



# In[ ]:




