from __future__ import print_function
from builtins import range

import sys
import numpy as np
import cv2
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
        #data = np.expand_dims(data,axis=0)
        plt.imshow(data)
    else:
        #for ch in range(3-data.shape[0]):
        #    data = np.concatenate([data,np.zeros((1,data.shape[1],data.shape[2]),dtype=np.uint8)])
        #data = data.transpose((1,2,0)) 
        print(data.shape)
        m = np.sqrt(data.ndim)
        rows= int(np.floor(m))
        cols= int(np.ceil(m))
        print(rows,cols)
        fig, axs = plt.subplots(rows,cols)
        fig.subplots_adjust(hspace = .5, wspace=.01)
        print(axs.shape)
        for n in range(data.ndim-1):
            print(n)
            axs[n].imshow(data[n])
            axs[n].set_title("dim="+str(n))
        #a= np.argmax(data, axis=0).astype(np.uint8)
        #print(a.shape)
        #plt.imshow(a)
        #img = np.zeros((a.shape[0],a.shape[1],3),dtype=np.uint8)
        #--------------------- TEST alg -------------
        #_, l_cont, l_hier = cv2.findContours(a, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #for i, l_cnt in enumerate(l_cont):
        #    if(l_cnt.shape[0] < 4):
        #        cv2.drawContours(img, [l_cnt], 0, (255,0,0), 3)
        #        print('l<4')
        #        #continue
        #    elif (cv2.contourArea(l_cnt) < 0.1*1024):
        #        cv2.drawContours(img, [l_cnt], 0, (0,255,0), 3)
        #        print('l<0.1')
        #       # continue
        #    elif (not cv2.isContourConvex(l_cnt)):
        #        l_cnt = cv2.convexHull(l_cnt)
        #        cv2.drawContours(img, [l_cnt], 0, (0,0,255), 3)
        #    else:
        #        cv2.drawContours(img, [l_cnt], 0, (0,0,255), 3)
        #    plt.imshow(img,cmap='hot')
        #    plt.show()
            


    #plt.imshow(data)
    plt.show()
    exit()

if __name__ == '__main__':
    if len(sys.argv) > 1 and  sys.argv[1] != '-h':
        main()
    else:
        print("Usage: python {} <pickle_file>".format(sys.argv[0]))
