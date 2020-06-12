import tensorflow as tf
import skimage.io as io
from skimage.transform import resize
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

if __name__=='__main__':

    with h5py.File('filter_org_img_25.hdf5', 'r') as hf:
        org_ip = hf['ip_data'][:]

    with h5py.File('filter_img_msk_25.hdf5', 'r') as hf:
        msk_ip = hf['op_data'][:]

    print(org_ip.shape)
    print(msk_ip.shape)

    failId = []
    for i in range(org_ip.shape[0]):

        img = msk_ip[i].reshape((256, 256))
        if img.sum() == 0:
            failId.append(i)

    print(len(failId))
    num = org_ip.shape[0] - len(failId)
    msk = np.zeros((num, 256, 256, 1), dtype=np.float32)
    org = np.zeros((num, 256, 256, 3), dtype=np.float32)

    index = 0
    for i in range(org_ip.shape[0]):

        if i in failId:
            continue
        else:
            msk[index] = msk_ip[i]
            org[index] = org_ip[i]
            index = index + 1

    print(msk.shape)

    DATA_NAME = 'filter_msk_20.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('op_data', data=msk)

    DATA_NAME = 'filter_org_20.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('ip_data', data=org)