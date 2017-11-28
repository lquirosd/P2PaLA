from __future__ import print_function
from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

try:
    import cPickle as pickle
except:
    import pickle #--- To handle data imports/export

#--- TODO: Add logging


class htrDataset(Dataset):
    """
    Class to handle HTR dataset feeding
    """
    def __init__(self,img_lst,label_lst, transform=None):
        """
        Args:
            img_lst (string): Path to the list of images to be processed
            label_lst (string): Path to the list of label files to be processed
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        #--- save all paths into a single dic
        self.img_paths = open(img_lst,'r').readlines()
        self.img_paths = [x.rstrip() for x in self.img_paths]
        self.label_paths = open(label_lst, 'r').readlines()
        self.label_paths = [x.rstrip() for x in self.label_paths]
        self.img_ids = [os.path.splitext(os.path.basename(x))[0] for x in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        #--- swap color axis because
        #--- cv2 image: H x W x C
        #--- torch image: C X H X W
        #---Keep arrays on float32 format for GPU compatibility
        image = (image.transpose((2,0,1))/255).astype(np.float32)
        fh = open(self.label_paths[idx],'r')
        label = pickle.load(fh)
        label = (label/255).astype(np.float32)
        fh.close()
        sample = {'image': image, 'label': label, 'id': self.img_ids[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample

class toTensor(object):
    """Convert dataset sample (ndarray) to tensor"""
    def __call__(self,sample):
        #--- TODO: check if its better to do not copy label array
        image, label = sample['image'], sample['label']
        return {'image':torch.from_numpy(image),
                'label':torch.from_numpy(label),
                'id':sample['id']}

class randomFlip(object):
    """randomly flip image in a sample"""
    def __init__(self,axis=1,prob=0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self,sample):
        if torch.rand(1)[0] < self.prob:
            image, label = sample['image'], sample['label']
            #--- TODO: Check why is a must to copy the array here
            #--- if not error raises: RuntimeError: some of the strides of a
            #---    given numpy array are negative. This is currently not 
            #---    supported, but will be added in future releases.
            image = np.flip(image, self.axis).copy()
            label = np.flip(label, self.axis).copy()
            return {'image': image, 'label': label, 'id':sample['id']}
        else:
            return sample

#--- TODO: Add normalize transform 

