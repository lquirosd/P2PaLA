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
import subprocess
import tempfile
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.optparse import Arguments as arguments
from page_xml.xmlPAGE import pageData
from evalTools import metrics as ev

#import matplotlib.pyplot as plt

def compute_metrics(hyp,target,opts,logger=None):
    """
    """
    #logger = logging.getLogger(__name__) if logger==None else logger 
    if logger==None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    num_samples = len(target)
    metrics = {}
    #--- force dict to appear in the same order always
    summary = OrderedDict()
    if opts.out_mode == 'L' or opts.out_mode == 'LR':
        metrics.update({'p_bl' : np.empty(num_samples,dtype=np.float),
                        'r_bl' : np.empty(num_samples,dtype=np.float),
                        'f1_bl': np.empty(num_samples,dtype=np.float)})
        t_file = '/tmp/'

        #--- sufix must be added because for some extrange reason
        #--- Transkribus tool need it 
        t_fd, t_path = tempfile.mkstemp(suffix='.lst')
        h_fd, h_path = tempfile.mkstemp(suffix='.lst')
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
            logger.debug(_err)
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
        summary['p_bl'] = bl_results[-6]
        summary['r_bl'] = bl_results[-5]
        summary['f1_bl'] = bl_results[-4]

    if opts.out_mode == 'R' or opts.out_mode == 'LR':
        metrics.update ({'p_acc' : np.empty(num_samples,dtype=np.float),
                         'm_acc' : np.empty(num_samples,dtype=np.float),
                         'm_iu'  : np.empty(num_samples,dtype=np.float),
                         'f_iu'  : np.empty(num_samples,dtype=np.float)})
        per_class_m = np.zeros((num_samples,np.unique(opts.regions_colors.values()).size+1),dtype=np.float)
        logger.info("-"*10 + "REGIONS EVALUATION RESULTS" + "-"*10)
        logger.debug("p_cc,m_acc,m_iu,f_iu,target_file,hyp_file")
        hyp = np.sort(hyp)
        target = np.sort(target)
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
            tmp = ev.per_class_accuraccy(hyp_mask,target_mask)
            for m,c in zip(tmp[0],tmp[1]):
                per_class_m[i,c] = m
            
            #per_class_m['pc_p_acc'][i] = ev.per_class_accuraccy(hyp_mask,target_mask)
            #per_class_m += ev.per_class_accuraccy(hyp_mask,target_mask)[0]  
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

        summary.update({m: metrics[m].sum() / num_samples for m in metrics})
        logger.info("Pixel accuracy:  {}".format(summary['p_acc']))
        logger.info("Mean accuracy:   {}".format(summary['m_acc']))
        logger.info("Mean IU:          {}".format(summary['m_iu']))
        logger.info("freq weighted IU: {}".format(summary['f_iu']))
        logger.info("Per_class Pixel accuracy: {}".format(per_class_m.sum(axis=0)/num_samples))
        mm=per_class_m.sum(axis=0)/num_samples
        #print("BG:{}".format(mm[0]))
        #for n,c in opts.regions_colors.items():
        #    print("{}:{}".format(n,mm[c]))
    #--- return averages only 
    return summary


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
