from __future__ import print_function
from __future__ import division
from builtins import range

import logging
import sys
import os
import time
import shutil
import numpy as np
import cv2
import errno
import gc
import subprocess
import tempfile

from utils.optparse import Arguments as arguments
from page_xml.xmlPAGE import pageData
from evalTools import metrics as ev
#import matplotlib.pyplot as plt

def compute_metrics(hyp,target,opts,logger=None):
    """
    """
    logger = logging.getLogger(__name__) if logger==None else logger 

    num_samples = len(target)
    metrics = {}
    if opts.out_mode == 'L' or opts.out_mode == 'LR':
        metrics.update({'p_bl' : np.empty(num_samples,dtype=np.float),
                        'r_bl' : np.empty(num_samples,dtype=np.float),
                        'f1_bl': np.empty(num_samples,dtype=np.float)})
        t_file = '/tmp/'

        #--- sufix must be added because for some extrange reason
        #--- Transkribus tool need it 
        t_fd, t_path = tempfile.mkstemp(suffix='.lst')
        h_fd, h_path = tempfile.mkstemp(suffix='.lst')
        print(t_path,h_path)
        try:
            with os.fdopen(t_fd, 'w') as tmp:
                tmp.write('\n'.join(target))
            with os.fdopen(h_fd, 'w') as tmp:
                tmp.write('\n'.join(hyp))
            evaltool = os.path.dirname(__file__) + '/baselineEvaluator.jar'
            cmd = subprocess.Popen(['java','-jar', evaltool,
                                    '-no_s', t_path, h_path],
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            bl_results, _err = cmd.communicate()
            print(_err)
        finally:
            os.remove(t_path)
            os.remove(h_path)

        logger.info("-"*10 + "BASELINE EVALUATION RESULTS" + "-"*10)
        logger.debug(bl_results)
        bl_results = bl_results.split('\n')
        for i in range(num_samples):
            res = bl_results[17+i].split(',')
            metrics['p_bl'][i] = float(res[0])
            metrics['r_bl'][i] = float(res[1])
            metrics['f1_bl'][i] = float(res[2])
        logger.info(bl_results[-6])
        logger.info(bl_results[-5])
        logger.info(bl_results[-4])

    if opts.out_mode == 'R' or opts.out_mode == 'LR':
        metrics.update ({'p_acc' : np.empty(num_samples,dtype=np.float),
                         'm_acc' : np.empty(num_samples,dtype=np.float),
                         'm_iu'  : np.empty(num_samples,dtype=np.float),
                         'f_iu'  : np.empty(num_samples,dtype=np.float)})
        logger.info("-"*10 + "REGIONS EVALUATION RESULTS" + "-"*10)
        logger.debug("p_cc,m_acc,m_iu,f_iu,target_file,hyp_file")
        for i,(h,t) in enumerate(zip(hyp,target)):
            target_data = pageData(t)
            target_data.parse()
            img_size = np.array(target_data.get_size())
            target_mask = target_data.build_mask(img_size,'TextRegion', opts.regions_colors)
            hyp_data = pageData(h)
            hyp_data.parse()
            hyp_mask = hyp_data.build_mask(img_size,'TextRegion', opts.regions_colors)

            metrics['p_acc'][i] = ev.pixel_accuraccy(hyp_mask,target_mask)
            metrics['m_acc'][i] = ev.mean_accuraccy(hyp_mask,target_mask)
            metrics['m_iu'][i]  = ev.mean_IU(hyp_mask,target_mask)
            metrics['f_iu'][i]  = ev.freq_weighted_IU(hyp_mask,target_mask)
            logger.debug('{:.4f},{:.4f},{:.4f} {:.4f},{},{}'.format(
                        metrics['p_acc'][i],
                        metrics['m_acc'][i],
                        metrics['m_iu'][i],
                        metrics['f_iu'][i],
                        t,h))

        #t_polygons = target_data.get_polygons('TextRegion')
        #h_polygons = hyp_data.get_polygons('TextRegion')
        #metrics['m_struct'][i] = ev.matching_structure(h_polygons,t_polygons)
        #ev.matching_structure(h_polygons,t_polygons)

    logger.info("Pixel accuraccy:  {}".format(metrics['p_acc'].sum()/num_samples))
    logger.info("Mean accuraccy:   {}".format(metrics['m_acc'].sum()/num_samples))
    logger.info("Mean IU:          {}".format(metrics['m_iu'].sum()/num_samples))
    logger.info("freq weighted IU: {}".format(metrics['f_iu'].sum()/num_samples))
    return metrics


def main():
    """
    """
    in_args = arguments()
    opts = in_args.parse()
    #--- check if bot
    with open(opts.target_list, 'r') as fh:
        target_list = fh.readlines()
    target_list = [x.rstrip() for x in target_list]

    with open(opts.hyp_list, 'r') as fh:
        hyp_list = fh.readlines()
    hyp_list = [x.rstrip() for x in hyp_list]

    if len(target_list) == len(hyp_list):
        compute_metrics(hyp_list,target_list,opts)
    else:
        print("ERROR")


if __name__ == '__main__':
    main()
