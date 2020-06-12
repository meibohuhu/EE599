import tensorflow as tf
import skimage.io as io
from skimage.transform import resize
import numpy as np
import h5py

if __name__=='__main__':

    # read data from hdf5 file:
    with h5py.File('filter_org_20.hdf5', 'r') as hf:
        train_input = hf['ip_data'][:]

    with h5py.File('filter_msk_20.hdf5', 'r') as hf:
        train_output = hf['op_data'][:]

    train_together = np.zeros((4800, 256, 256, 4), dtype=np.float32)

    train_output_part = train_output[:4800, :, :, :]
    train_output_part = train_output_part.reshape(4800, 256, 256)
    train_together[:, :, :, :3] = train_input[:4800, :, :, :]
    train_together[:, :, :, -1] = train_output_part

    # build dataGenerator to do data augment
    augment_data = np.zeros((9600, 256, 256, 4), dtype=np.float32)

    # rotate the images to 90 degree
    for i in range(67):

        batch = train_together[100 * i:100 * (i + 1), :, :, :]
        rot_90 = tf.image.rot90(batch, k=1)
        rot_90_array = np.array(rot_90)
        augment_data[100 * i:100 * (i + 1), :, :, :] = rot_90_array
        print('batch ' + str(i+1) + ' saved. 67 in total.')

    # save the augment data
    augment_data_input = augment_data[:, :, :, :3]
    augment_data_output = augment_data[:, :, :, -1]
    augment_data_output = augment_data_output.reshape(6700, 256, 256, 1)

    new_ipdata = np.concatenate((train_input, augment_data_input), axis=0)
    new_opdata = np.concatenate((train_output, augment_data_output), axis=0)


    DATA_NAME = 'filter_img_msk_aug_20.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('op_data', data=new_opdata)

    DATA_NAME = 'filter_org_img_aug_20.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('ip_data', data=new_ipdata)

    DATA_NAME = 'augment_msk_25.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('aug_msk', data=augment_data_output)

    DATA_NAME = 'augment_org_25.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('aug_org', data=augment_data_input)