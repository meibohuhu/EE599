import tensorflow as tf
from pycocotools.coco import COCO
import skimage.io as io
from skimage.transform import resize
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

if __name__=='__main__':

    DEBUG = False

    # create directory to build coco object.
    dataDir = '../Data/annotations'
    dataType = 'train2017'  # CAN CHANGE FOR TRAIN AND VAL
    annFile = dataDir + '/instances_{}.json'.format(dataType)

    # build coco object.
    coco = COCO(annFile)

    # get all images containing given categories, select one at random.
    catIds = coco.getCatIds(catNms=['person'])  # get category id
    imgIds = coco.getImgIds(catIds=catIds)  # get image id
    img = coco.loadImgs(imgIds)  # load all satisfied images: a list of dictionary.

    # filter the data
    input_img_ids = []
    img_num = len(imgIds)
    msk = np.zeros((img_num, 256, 256, 1), dtype=np.float32)
    org = np.zeros((img_num, 256, 256, 3), dtype=np.float32)
    index = 0

    if DEBUG:
        img_num = 10

    for i in range(20):

        print('round ' + str(i) + ', total ' + str(len(imgIds)))

        input_img = io.imread(img[i]['coco_url'])

        # data clean -- filter the grey images
        if input_img.ndim != 3 or input_img.shape[2] != 3:
            continue

        img_id = img[i]['id']
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # filter the single person
        if len(anns) != 1:
            continue

        binmsk = coco.annToMask(anns[0])

        # compute threshold.
        mask_area = binmsk.sum()
        total_area = binmsk.shape[0] * binmsk.shape[1]

        if mask_area < 0.20 * total_area:
            continue

        # save the org_img, msk and ids
        input_img_ids.append(img_id)
        print('image ' + str(img_id) + ' appended normally.')

        # save the org_img
        I_img = resize(input_img, (256, 256, 3), mode='constant', preserve_range=True)
        I_img = I_img / 255.0
        org[index] = I_img
        '''
        if len(input_img_ids) == 1 or len(input_img_ids) == 5:
            plt.axis('off')
            plt.imshow(I_img)
            plt.savefig('rs_org_{}.png'.format(img_id))
        
        '''


        # save the msk img
        bkr_img = resize(binmsk, (256, 256, 1), mode='constant', preserve_range=True)
        msk[index] = bkr_img
        index = index + 1
        '''
        if len(input_img_ids) == 1 or len(input_img_ids) == 5:
            in_img = bkr_img.reshape((256, 256))
            plt.axis('off')
            plt.imshow(in_img)
            plt.savefig('rs_bkr_{}.png'.format(img_id))
        
        '''

    # save data in hdf5
    size = len(input_img_ids)
    print(size)

    filter_img_id = np.array(input_img_ids)
    msk_x = msk[:size, :, :, :]
    org_x = org[:size, :, :, :]
    # save the data id
    DATA_NAME = 'filter_img_ids_25.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('ids', data=filter_img_id)

    DATA_NAME = 'filter_img_msk_25.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('op_data', data=msk_x)

    DATA_NAME = 'filter_org_img_25.hdf5'
    with h5py.File(DATA_NAME, 'w') as hf:
        hf.create_dataset('ip_data', data=org_x)