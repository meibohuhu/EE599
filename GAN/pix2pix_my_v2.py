#!/usr/bin/env python
# coding: utf-8

# In[6]:


# imoport 
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
import h5py
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
# import tensorflow_addons as tfa
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
import os
import time
from matplotlib import pyplot as plt
# from IPython import display


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# settings
Config ={}
Config['ROOT_PATH_INPUT']='/home/ec2-user/final/filter_org_img_25.hdf5'
Config['ROOT_PATH_MASK']='/home/ec2-user/final/filter_img_msk_25.hdf5'
Config['BUFFER_SIZE']=400
Config['IMG_WIDTH'] =256
Config['IMG_HEIGHT'] =256
# Config['random_flip'] =True
Config['epochs'] =5
Config['lr'] =0.002
Config['BATCH_SIZE'] =2
Config['LAMBDA'] =1.5


# In[ ]:





# In[ ]:


# def normalize(input_image, real_image):
#     input_image = (input_image) - 1
#     real_image = (real_image) - 1

#     return input_image, real_image


# In[ ]:


with h5py.File(Config['ROOT_PATH_INPUT'], 'r') as hf:
    train_input = hf['ip_data'][:]
    
with h5py.File(Config['ROOT_PATH_MASK'], 'r') as hf:
    train_output = hf['op_data'][:]
    
train_dataset = tf.data.Dataset.from_tensor_slices((train_input,train_output))
# for element in dataset: 
#     print(element) 

# train_dataset = train_dataset.map(load_image_train,
#                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(Config['BUFFER_SIZE'])
train_dataset = train_dataset.batch(Config['BATCH_SIZE'])


# In[ ]:


test_input = train_input[-6,:,:,:]
test_output = train_output[-6,:,:,:]
test_dataset = tf.data.Dataset.from_tensor_slices((test_input,test_output))
test_dataset = test_dataset.batch(Config['BATCH_SIZE'])


# In[ ]:


# def G
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def Generator(n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""
    # Contracting Path
    input_img = Input((256, 256, 3), name='img')

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


# In[ ]:


generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
# gen_output = generator(inp[tf.newaxis,...], training=False)


# In[ ]:


# show the output 
# plt.imshow(gen_output[0,...])


# In[ ]:


# def G loss

def IOU_calc(y_true, y_pred):
#     if K.max(y_true) == 0.0:
#         return IOU_calc(1-y_true, 1-y_pred)
    smooth =1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # union = K.sum(K.sign(y_true_f+y_pred_f))
    #union = K.sum(y_true_f)+ K.sum(y_pred_f)-intersection
    union = K.sum(y_true_f)+ K.sum(y_pred_f)

    return (intersection + smooth) / (union + smooth)


def IOU_calc_loss(y_true, y_pred):
    return 1.-IOU_calc(y_true, y_pred)

def weighted_bce(y_true, y_pred):
    weights = (y_true * 59.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def my_loss(y_true, y_pred):
    loss = K.binary_crossentropy(y_true,y_pred)*0.0001 + IOU_calc_loss(y_true, y_pred)
#     loss = K.binary_crossentropy(y_true,y_pred)+ tfa.losses.SigmoidFocalCrossEntropy()*0.5
#     loss = tfa.losses.GIoULoss(mode='iou')
#       loss = tfa.losses.sigmoid_focal_crossentropy()
    return loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # my_loss
    opt_loss = my_loss(target, gen_output)

    total_gen_loss = gan_loss + (Config['LAMBDA'] * opt_loss)

    return total_gen_loss, gan_loss, opt_loss


# In[ ]:


# def D
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

def Discriminator():
    
    layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    n_pool=5
    growth_rate=16
    
    input = Input(shape=(256, 256, 3))
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

    outputs = Dense(84,activation='sigmoid')(t)
    
    model = Model(inputs=input, outputs=outputs)
    return model


# In[ ]:


discriminator = Discriminator()
# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)


# In[ ]:


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# In[ ]:


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# In[ ]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# In[ ]:


# def generate_images(model, test_input, tar):
#     prediction = model(test_input, training=True)
# #     plt.figure(figsize=(15,15))


# In[ ]:


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# In[ ]:


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, opt_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

#     with summary_writer.as_default():
#         tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
#         tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
#         tf.summary.scalar('opt_loss', opt_loss, step=epoch)
#         tf.summary.scalar('disc_loss', disc_loss, step=epoch)


# In[ ]:


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

#     display.clear_output(wait=True)

#     for example_input, example_target in test_ds.take(1):
#       generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)


# In[ ]:


fit(train_dataset, Config['epochs'])


# In[ ]:


generator.save('g_model.h5')
discriminator.save('d_model.h5')


# In[2]:


# import tensorflow as tf
# import h5py

# with h5py.File('/Users/wangjiamian/Desktop/USC/EE599/final_project/COCO/tr_6K_LYC_ori_mask/filter_img_msk_25.hdf5', 'r') as hf:
#     train_output = hf['op_data'][:]
# dataset = tf.data.Dataset.from_tensor_slices(train_output) 
# for element in dataset: 
#     print(element) 


# In[3]:





# In[4]:


# # restoring the latest checkpoint in checkpoint_dir
# checkpoint.restore('/Users/wangjiamian/Desktop/USC/EE599')
# for inp, tar in test_dataset.take(6):
#     results = generate_images(generator, inp, tar)


# In[ ]:




