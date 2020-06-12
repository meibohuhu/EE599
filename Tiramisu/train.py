#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[5]:


## #!/usr/bin/python

# from plot_loss_image import PlotLosses
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_2 import my_model
import h5py
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.models as models
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
import tensorflow_addons as tfa

# from utils import ExpDecay, WeightedCrossentropy, MapillaryGenerator

Config ={}
Config['crop_width'] =256
Config['crop_height'] =256
Config['random_flip'] =True
Config['epochs'] =25
Config['lr'] =0.01
Config['batch_size'] =2
# Config['decay'] =0.995
# Config['n_gpu'] =1
# Config['n_cpu'] =8
# Config['weights'] =None



#### Train ####
def IOU_calc(y_true, y_pred):
#     if K.max(y_true) == 0.0:
#         return IOU_calc(1-y_true, 1-y_pred)
    smooth =1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # union = K.sum(K.sign(y_true_f+y_pred_f))
    union = K.sum(y_true_f)+ K.sum(y_pred_f)-intersection
    return (intersection + smooth) / (union + smooth)


def IOU_calc_loss(y_true, y_pred):
    return 1.-IOU_calc(y_true, y_pred)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.mean(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
    return focal_loss_fixed

def weighted_bce(y_true, y_pred):
    weights = (y_true * 6.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def my_loss(y_true, y_pred):
#     loss = K.binary_crossentropy(y_true,y_pred)*0.1 + IOU_calc_loss(y_true, y_pred)
#     loss = K.binary_crossentropy(y_true,y_pred)+ tfa.losses.SigmoidFocalCrossEntropy()*0.5
#     loss = tfa.losses.GIoULoss(mode='iou')
#     loss = tfa.losses.sigmoid_focal_crossentropy()
    loss = weighted_bce(y_true, y_pred)*0.1 + IOU_calc_loss(y_true, y_pred)

    return loss


# load data: train/val/test
with h5py.File('/home/ec2-user/project/filter_org_20.hdf5', 'r') as hf:
    train_input = hf['ip_data'][:]
with h5py.File('/home/ec2-user/project/filter_msk_20.hdf5', 'r') as hf:
    train_output = hf['op_data'][:]

# Optimizer
optim = optimizers.SGD(lr=Config['lr'], momentum=0.9)

# Model
layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
tiramisu = my_model(Config['crop_width'], Config['crop_height'],layer_per_block)
# tiramisu = make_parallel(tiramisu, Config['n_gpu'])

# test model using VGG16

# Training
# tiramisu.compile(optim, loss="binary_crossentropy", metrics=['accuracy'])
tiramisu.compile(optim, loss=my_loss, metrics=[IOU_calc,'accuracy'])

tiramisu.summary()

# need change

results = tiramisu.fit(train_input, train_output, batch_size = Config['batch_size'], 
             validation_split=0.2, shuffle=True, epochs=Config['epochs'])

# save model
tiramisu.save('tiramisu_epo20_lr012_WBCE.h5')  

# plot model 
tf.keras.utils.plot_model(tiramisu, to_file='tiramisu-model.png', dpi=256)

# draw and save plot
# loss = results.history['loss']
# val_loss = results.history['val_loss']

# iou = results.history['mean_io_u']
# val_iou = results.history['val_mean_io_u']

# epochs = np.arange(0,Config['epochs'])

# plt.figure()
# plt.plot(epochs, iou, label='mean_io_u')
# plt.plot(epochs, val_iou, label='val_mean_io_u')
# plt.xlabel('epochs')
# plt.ylabel('IoU')
# plt.legend()
# plt.savefig('learning_IoU.png', dpi=256)


    
###############


# In[3]:


# from tensorflow.keras import backend as K
# import tensorflow as tf
# x = [0,1,2,1]
# a = [1,0,1,0]
# z = tf.convert_to_tensor(x) *tf.convert_to_tensor(a)
# print(z)
# y = K.sign(x)
# print(y)
# print(1.-z)


# In[ ]:




