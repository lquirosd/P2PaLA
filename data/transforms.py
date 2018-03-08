from __future__ import print_function
from __future__ import division
from builtins import range

#import os
import math

import numpy as np
import torch
#from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
#import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter
#import logging

def build_transforms(opts,train=True):
    tr = []
    if train:
        #--- add flip transformation
        if opts.flip_img:
            tr.append(randomFlip(axis=1,prob=opts.trans_prob))
        #--- add affine transformation
        if opts.affine_trans:
            tr.append(affine(prob=opts.trans_prob,t_stdv=opts.t_stdv,
                             r_kappa=opts.r_kappa,sc_stdv=opts.sc_stdv,
                             sh_kappa=opts.sh_kappa))
        #--- add elastic deformation
        if opts.elastic_def:
            tr.append(elastic(prob=opts.trans_prob,alpha=opts.e_alpha, 
                              stdv=opts.e_stdv))
    
    #--- transform data(ndarrays) to tensor
    tr.append(toTensor())
    #--- Normalize to 0-mean, 1-var
    tr.append(normalizeTensor(mean=None,std=None))

    #--- add all trans to que tranf queue
    return tv_transforms.Compose(tr)

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

class normalizeTensor(object):
    """Normalize tensor to given meand and std, or its mean and std per 
    channel"""
    def __init__(self,mean=None,std=None):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        if torch.is_tensor(sample['image']): 
            if self.mean is None or self.std is None:
                self.mean = []
                self.std = []
                for t in sample['image']:
                    self.mean.append(t.mean())
                    self.std.append(t.std())
            if not len(self.mean) == sample['image'].shape[0] or not len(self.std) == sample['image'].shape[0]:
                raise ValueError('mean and std size must be equal to the number of channels of the input tensor.'
                        )
            for i,t in enumerate(sample['image']):
                t.sub_(self.mean[i]).div_(self.std[i])
        else:
            raise TypeError('Input image is not a tensor, make sure to queue this after toTensor transform')
        return sample

class normalizeArray(object):
    """Normalize array to given meand and std, or its mean and std per 
    channel"""
    def __init__(self,mean=None,std=None):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        if type(sample['image']) is np.ndarray: 
            if self.mean is None or self.std is None:
                self.mean = []
                self.std = []
                for t in sample['image']:
                    self.mean.append(t.mean())
                    self.std.append(t.std())
            if not len(self.mean) == sample['image'].shape[0] or not len(self.std) == sample['image'].shape[0]:
                raise ValueError('mean and std size must be equal to the number of channels of the input array.'
                        )
            for i,t in enumerate(sample['image']):
                t.add(-self.mean[i]).divide(self.std[i])
        else:
            raise TypeError('Input image is not a ndarray, make sure to queue this before toTensor transform')
        return sample

class elastic(object):
    """
    Elastric deformation over an array.
    Based on:
    @inproceedings{Simard03,
        author = {Simard, Patrice Y. and Steinkraus, Dave and Platt, John},
        title = {Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis},
        booktitle = {},
        year = {2003},
        month = {August},
        publisher = {Institute of Electrical and Electronics Engineers, Inc.},
        url = {https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/},
    }
    """
    def __init__(self,alpha=34, stdv=4,prob=0.5):
        self.alpha = alpha
        self.stdv = stdv
        self.prob = prob

    def __call__(self,sample):
        """
        """
        if torch.rand(1)[0] < self.prob:
            shape = sample['image'][0].shape
            dmin = min(shape)
            #dx = gaussian_filter((self.rnd.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            #dy = gaussian_filter((self.rnd.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            #--- if stdv is too small (0.0x) the displacement field looks random, and if stdv is too large (stdv > 8)
            #--- the displacement field looks like translations. Since normally we perform translations as affine transf
            #--- a intermediate value of stdv will suffice, from [Simard03] stdv=4 is a good option.
            dx = gaussian_filter((torch.rand(shape) * 2 - 1), self.stdv, mode="constant", cval=0) * dmin * self.alpha
            dy = gaussian_filter((torch.rand(shape) * 2 - 1), self.stdv, mode="constant", cval=0) * dmin * self.alpha
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
            if np.ndim(sample['image']) > 2:
                for t in sample['image']:
                    t[:] = map_coordinates(t, indices, order=1).reshape(shape)
            else:
                sample['image'] = map_coordinates(sample['image'], indices, order=1).reshape(shape)
            if np.ndim(sample['label']) > 2:
                for t in sample['label']:
                    t[:] = map_coordinates(t, indices, order=1).reshape(shape)
            else:
                sample['label'] = map_coordinates(sample['label'], indices, order=1).reshape(shape)

        return sample

class affine(object):
    """
    Perform affine transformations over the input array
    [trnslation, rotation, shear, scale]
    """
    def __init__(self,prob=0.5,t_stdv=0.02,r_kappa=30,sc_stdv=0.12,sh_kappa=20):
        self.prob    = prob
        self.t_stdv  = t_stdv
        self.r_kappa  = r_kappa
        self.sh_kappa = sh_kappa
        self.sc_stdv = sc_stdv

    def __call__(self,sample):
        #--- transf must follow this order:
        #--- translate -> rotate -> shear -> scale
        ch,H,W = sample['image'].shape
        #--- centering mat
        C,Cm= np.eye(3),np.eye(3)
        C[0,2] = W/2; C[1,2] = H/2
        Cm[0,2] = -W/2; Cm[1,2] = -H/2
        T = np.eye(3,3)

        #--- Translate:
        if np.random.rand() < self.prob:
            #--- normal distribution is used to translate the data, an small 
            #--- stdv is recomended in order to keep the data inside the image
            #--- [0.001 < stdv < 0.02 is recomended]
            T[0:2,2] = np.random.rand(2)*[W,H] * self.t_stdv
        #--- rotate
        if torch.rand(1)[0] < self.prob:
        #--- r_kappa value controls von mises "concentration", so to kepp 
        #--- the rotation under controlled parameters r_kappa=30 keeps
        #--- theta around +-pi/8 (if mu=0.0)
            D = np.eye(3)
            theta = np.random.vonmises(0.0,self.r_kappa)
            D[0:2,0:2] = [[math.cos(theta),math.sin(theta)],
                          [-math.sin(theta),math.cos(theta)]]
            T = np.dot(np.dot(np.dot(T,C),D),Cm)
        #--- Shear (vert and Horz)
        if np.random.rand() < self.prob:
            #--- under -pi/8 < theta < pi/8 range tan(theta) ~ theta, then 
            #--- computation of tan(theta) is ignored. kappa will be 
            #--- selected to comply this restriction [kappa ~> 20 is a good value]
            theta = np.random.vonmises(0.0,self.sh_kappa)
            D = np.eye(3)
            D[0,1] = theta
            T = np.dot(np.dot(np.dot(T,C),D),Cm)
        if np.random.rand() < self.prob:
            theta = np.random.vonmises(0.0,self.sh_kappa)
            D = np.eye(3)
            D[1,0] = theta
            T = np.dot(np.dot(np.dot(T,C),D),Cm)
        #--- scale
        if np.random.rand() < self.prob:
            #--- Use log_normal distribution with mu=0.0 to perform scale, 
            #--- since scale factor must be > 0, stdv is used to control the 
            #--- deviation from 1 [0.1 < stdv < 0.5 is recomended]
            D = np.eye(3)
            D[0,0],D[1,1] = np.exp(np.random.rand(2)*self.sc_stdv)
            T = np.dot(np.dot(np.dot(T,C),D),Cm)

        if (T == np.eye(3)).all():
            return sample
        else:
            if np.ndim(sample['image']) > 2:
                for t in sample['image']:
                    t[:] = affine_transform(t,T)
            else:
                sample['image'] = affine_transform(sample['image'],T)
            if np.ndim(sample['label']) > 2:
                for t in sample['label']:
                    t[:] = affine_transform(t,T)
            else:
                sample['label'] = affine_transform(sample['label'],T)
            
            return sample

