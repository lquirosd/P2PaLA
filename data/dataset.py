from __future__ import print_function
from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import Dataset
#from torchvision import transforms
import cv2
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.filters import gaussian_filter
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
            #--- pre-compute per class weigths 
            #--- one count is added per class in order to avoid zero prob.
            #--- weights will be restrict to the interval [1/log(1+c), 1/log(c)]
            #--- for the default c=1.02 => [50.49,1.42]
            if opts.out_mode == 'L':
                self.w = np.ones(2,dtype=np.float)
                for l in self.label_paths:
                    with open(l,'r') as fh:
                        label = pickle.load(fh)
                    self.w += np.bincount(label.flatten(),minlength=2)
                self.w = self.w/((len(self.label_paths) * opts.img_size.sum()) + 2)
                self.w = 1/np.log(opts.weight_const + self.w)
            if opts.out_mode == 'LR':
                self.w = [np.ones(2,dtype=np.float), 
                          np.ones(len(opts.regions) + 1,dtype=np.float)]
                for l in self.label_paths:
                    with open(l,'r') as fh:
                        label = pickle.load(fh)
                    self.w[0] += np.bincount(label[0].flatten(),minlength=2)
                    self.w[1] += np.bincount(label[1].flatten(),minlength=len(opts.regions)+1)
                self.w[0] = self.w[0]/((len(self.label_paths) * opts.img_size.sum()) + 2)
                self.w[1] = self.w[1]/((len(self.label_paths) * opts.img_size.sum()) + len(opts.regions) + 1)
                self.w[0] = 1/np.log(opts.weight_const + self.w[0])
                self.w[1] = 1/np.log(opts.weight_const + self.w[1])
            if opts.out_mode == 'R':
                self.w = np.ones(len(opts.regions) + 1,dtype=np.float)
                for l in self.label_paths:
                    with open(l,'r') as fh:
                        label = pickle.load(fh)
                    self.w += np.bincount(label.flatten(),minlength=len(opts.regions))
                self.w = self.w/((len(self.label_paths) * opts.img_size.sum()) + len(opts.regions) + 1)
                self.w = 1/np.log(opts.weight_const + self.w)

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
        #--- TODO: Move norm comp and transforms to GPU
        image = (((2/255)*image.transpose((2,0,1)))-1).astype(np.float32)
        if self.build_label:
            fh = open(self.label_paths[idx],'r')
            label = pickle.load(fh)
            #--- TODO: change to opts.net+out_type == C
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

#class toTensor(object):
#    """Convert dataset sample (ndarray) to tensor"""
#    def __call__(self,sample):
#        for k,v in sample.iteritems():
#            if type(v) is np.ndarray:
#                #--- by default float arrays will be converted to float tensors
#                #--- and int arrays to long tensor.
#                sample[k] = torch.from_numpy(v)
#        return sample
#
#class randomFlip(object):
#    """randomly flip image in a sample"""
#    def __init__(self,axis=1,prob=0.5):
#        self.axis = axis
#        self.prob = prob
#
#    def __call__(self,sample):
#        if torch.rand(1)[0] < self.prob:
#            #--- TODO: Check why is a must to copy the array here
#            #--- if not error raises: RuntimeError: some of the strides of a
#            #---    given numpy array are negative. This is currently not 
#            #---    supported, but will be added in future releases.
#            for k,v in sample.iteritems():
#                if type(v) is np.ndarray:
#                    sample[k] = np.flip(v, self.axis).copy()
#            return sample
#        else:
#            return sample
#
##--- TODO: Add normalize transform 
#class normalizeTensor(object):
#    """Normalize image to given meand and std"""
#    def __init__(self,mean=0,std=1):
#        self.mean = mean
#        self.std = std
#
#    def __call__(self,sample):
#        mean = []
#        std = []
#        if torch.is_tensor(sample['image']): 
#            for t in sample['image']:
#                mean.append(t.mean())
#                std.append(t.std())
#            for i,t in enumerate(sample['image']):
#                t.sub_(mean[i]).div_(std[i])
#        else:
#            raise TypeError('Input image is not a tensor, make sure to queue this after toTensor transform')
#        return sample
#
#class elastic_transform(object):
#    """
#    Elastric transformation over the image.
#    Based on:
#    @inproceedings{Simard03,
#        author = {Simard, Patrice Y. and Steinkraus, Dave and Platt, John},
#        title = {Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis},
#        booktitle = {},
#        year = {2003},
#        month = {August},
#        publisher = {Institute of Electrical and Electronics Engineers, Inc.},
#        url = {https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/},
#    }
#    """
#    def __init__(self,alpha=34, sigma=4,prob=0.5):
#        self.alpha = alpha
#        self.sigma = sigma
#        self.prob = prob
#        self.rnd = None
#
#    def __call__(self,sample):
#
#        if torch.rand(1)[0] < self.prob:
#            if self.rnd is None:
#                self.rnd = np.random.RandomState(None)
#            shape = sample['image'][0].shape
#            dx = gaussian_filter((self.rnd.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
#            dy = gaussian_filter((self.rnd.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
#            #dx = gaussian_filter((torch.rand(shape) * 2 - 1).numpy(), self.sigma, mode="constant", cval=0) * self.alpha
#            #dy = gaussian_filter((torch.rand(shape) * 2 - 1).numpy(), self.sigma, mode="constant", cval=0) * self.alpha
#            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
#            sample['image'][0] = map_coordinates(sample['image'][0], indices, order=1).reshape(shape)
#            sample['image'][1] = map_coordinates(sample['image'][1], indices, order=1).reshape(shape)
#            sample['image'][2] = map_coordinates(sample['image'][2], indices, order=1).reshape(shape)
#            sample['label']= map_coordinates(sample['label'], indices, order=1).reshape(shape)
#
#        return sample
