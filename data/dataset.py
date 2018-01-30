from __future__ import print_function
from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import logging

try:
    import cPickle as pickle
except:
    import pickle #--- To handle data imports/export

class htrDataset(Dataset):
    """
    Class to handle HTR dataset feeding
    """
    def __init__(self,img_lst,label_lst=None, transform=None, logger=None, opts=None):
        """
        Args:
            img_lst (string): Path to the list of images to be processed
            label_lst (string): Path to the list of label files to be processed
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform
        #--- save all paths into a single dic
        self.img_paths = open(img_lst,'r').readlines()
        self.img_paths = [x.rstrip() for x in self.img_paths]
        self.build_label = False
        #--- Labels will be loaded only if label_lst exists
        if label_lst != None:
            self.label_paths = open(label_lst, 'r').readlines()
            self.label_paths = [x.rstrip() for x in self.label_paths]
            self.build_label = True
        self.img_ids = [os.path.splitext(os.path.basename(x))[0] for x in self.img_paths]
        self.opts = opts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        #--- swap color axis because
        #--- cv2 image: H x W x C
        #--- torch image: C X H X W
        #---Keep arrays on float32 format for GPU compatibility
        #--- Normalize to [-1,1] range
        image = (((2/255)*image.transpose((2,0,1)))-1).astype(np.float32)
        if self.build_label:
            fh = open(self.label_paths[idx],'r')
            label = pickle.load(fh)
            if self.opts.do_class:
                #--- convert labels to np.int for compatibility to NLLLoss
                label = label.astype(np.int)
            else:
                #--- norm to [-1,1] 
                label = (((2/255)*label)-1).astype(np.float32)
                #--- force array to be a 3D tensor as needed by conv2d
                if label.ndim == 2:
                    label = np.expand_dims(label, 0)
            fh.close()
            sample = {'image': image, 'label': label, 'id': self.img_ids[idx]}
        else:
            sample = {'image':image, 'id':self.img_ids[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample

class toTensor(object):
    """Convert dataset sample (ndarray) to tensor"""
    def __call__(self,sample):
        for k,v in sample.iteritems():
            if type(v) is np.ndarray:
                #--- by default float arrays will be converted to float tensors
                #--- and int arrays to long tensor.
                sample[k] = torch.from_numpy(v)
        return sample

class randomFlip(object):
    """randomly flip image in a sample"""
    def __init__(self,axis=1,prob=0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self,sample):
        if torch.rand(1)[0] < self.prob:
            #--- TODO: Check why is a must to copy the array here
            #--- if not error raises: RuntimeError: some of the strides of a
            #---    given numpy array are negative. This is currently not 
            #---    supported, but will be added in future releases.
            for k,v in sample.iteritems():
                if type(v) is np.ndarray:
                    sample[k] = np.flip(v, self.axis).copy()
            return sample
        else:
            return sample

#--- TODO: Add normalize transform 

