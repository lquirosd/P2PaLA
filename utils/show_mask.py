from __future__ import print_function
import sys
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle as pickle

import matplotlib.pyplot as plt

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    with open(sys.argv[1],'r') as fh:
        data = pickle.load(fh)
    if data.ndim == 2:
        data = np.expand_dims(data,axis=0)
    for ch in xrange(3-data.shape[0]):
        data = np.concatenate([data,np.zeros((1,data.shape[1],data.shape[2]),dtype=np.uint8)])
    data = data.transpose((1,2,0)) 
    plt.imshow(data)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1 and  sys.argv[1] != '-h':
        main()
    else:
        print("Usage: python {} <pickle_file>".format(sys.argv[0]))
