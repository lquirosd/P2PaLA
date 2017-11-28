from __future__ import print_function
from __future__ import division

import os
import glob
import logging

import numpy as np
import cv2
from multiprocessing import Pool
import itertools
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export

from page_xml.xmlPAGE import pageData


#---- Resize images
#---- PAGE processing
#---- Save img_lst and label_lst pointers 
#--- TODO: add logging

def _processData(params):
    """
    Resize image and extract mask from PAGE file 
    """
    (img_path,out_size,out_folder,classes,line_width,line_color) = params
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img_dir = os.path.dirname(img_path)
    if (os.path.isfile(img_dir + '/page/' + img_id + '.xml')):
        xml_path = img_dir + '/page/' + img_id + '.xml'
    else:
        logging.critical('No xml found for file {}'.format(img_path))
        raise Exception("Execution stop due Critical Errors")
    
    img_data = cv2.imread(img_path)
    #--- resize image 
    res_img = cv2.resize(img_data,(out_size[1],
                         out_size[0]),
                         interpolation=cv2.INTER_CUBIC)
    new_img_path = out_folder+'/'+img_id+'.png'
    cv2.imwrite(new_img_path,res_img)
    #--- get label
    gt_data = pageData(xml_path)
    reg_mask = gt_data.buildMask(out_size,'TextRegion', classes)
    lin_mask = gt_data.buildBaselineMask(out_size,line_color,line_width)
    label = np.array((lin_mask,reg_mask))
    new_label_path = out_folder + '/' + img_id + '.pickle'
    fh = open(new_label_path,'w')
    pickle.dump(label,fh,-1)
    fh.close()
    return (new_img_path, new_label_path)

def htrDataProcess(data_pointer, out_size, out_folder, classes,
                   line_width=10, line_color=128, processes=2):
    """ function to proces all data into a htr dataset"""
    formats = ['tif','tiff', 'png', 'jpg', 'jpeg','bmp']
    img_fh = open(out_folder + '/img.lst','w')
    label_fh = open(out_folder + '/label.lst','w')
    img_list = []
    for ext in formats:
        img_list.extend(glob.glob(data_pointer + '/*.' + ext))
    processed_data = []
    try:
        pool = Pool(processes=processes) #--- call without parameters = Pool(processes=cpu_count())
        l_list = len(img_list)
        params = itertools.izip(img_list,[out_size]*l_list,
                                   [out_folder]*l_list,
                                   [classes]*l_list,
                                   [line_width]*l_list,
                                   [line_color]*l_list)
        processed_data = pool.map(_processData,params)
    except Exception as e:
        pool.close()
        pool.terminate()
        print(e)
    else:
        pool.close()
        pool.join()
    processed_data = np.array(processed_data)
    np.savetxt(out_folder + '/img.lst',processed_data[:,0],fmt='%s')
    np.savetxt(out_folder + '/label.lst',processed_data[:,1],fmt='%s')
    return (out_folder + '/img.lst',out_folder + '/label.lst')
 
