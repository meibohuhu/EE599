import tensorflow as tf
from unet_model_msk import get_unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
# from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import MeanIoU
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt


if __name__=='__main__':

    def IOU_calc(y_true, y_pred):
        #     if K.max(y_true) == 0.0:
        #         return IOU_calc(1-y_true, 1-y_pred)
        smooth = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        # union = K.sum(K.sign(y_true_f+y_pred_f))
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return 2 * (intersection + smooth) / (union + smooth)

    # use mixed loss, IOU + MSE
    def IOU_calc_loss(y_true, y_pred):
        return 1-IOU_calc(y_true, y_pred)

    def my_loss(y_true, y_pred):
        loss = 0.1 * K.binary_crossentropy(y_true, y_pred) + IOU_calc_loss(y_true, y_pred)
        return loss

    # read data from hdf5 file:
    with h5py.File('train_org_20.hdf5', 'r') as hf:
        train_input = hf['ip_data'][:]

    with h5py.File('train_msk_20.hdf5', 'r') as hf:
        train_output = hf['op_data'][:]

    print(train_input.shape)
    print(train_output.shape)
    # import model
    epoch = 30
    model = get_unet()
    model.compile(optimizer=Adam(), loss=my_loss, metrics=[IOU_calc, 'accuracy']) # "binary_crossentropy"
    model.summary()

    # training
    H = model.fit(train_input, train_output, batch_size=32, validation_split=0.2, shuffle=True, epochs=epoch)

    ## provide some reports on the training
    # plot_model(model, to_file='train.png', show_shapes=True, show_layer_names=True)
    model.save('Train_img_model.hdf5')

    plt.figure()

    # for i in np.arange(0, N):
    plt.plot(np.arange(0, epoch), H.history["IOU_calc"], label="Train_IoU")
    plt.plot(np.arange(0, epoch), H.history["loss"], label="Train_loss")
    plt.plot(np.arange(0, epoch), H.history["val_IOU_calc"], label="Val_IoU")
    plt.plot(np.arange(0, epoch), H.history["val_loss"], label="Val_loss")
    plt.plot(np.arange(0, epoch), H.history["accuracy"], label="Train_acc")
    plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="Val_acc")

    plt.title("Training Loss/accuracy/iou on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy/IOU")
    plt.legend(loc='best')
    plt.savefig('Train_learning_curve.png')