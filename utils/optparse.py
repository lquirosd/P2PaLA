from __future__ import print_function
from __future__ import division

import logging
import numpy as np
from collections import OrderedDict
import argparse 
import os
from math import log
import multiprocessing

class arguments(object):
    """
    """
    def __init__(self):
        """
        """
        parser_description = """
        NN Implentation for Layout Analysis
        """
        regions = ['$tip','$par','$not','$nop','$pag']
        merge_regions = {'$par':['$pac']}
        n_cpus = multiprocessing.cpu_count()
        baseline_evaluator = '/home/lquirosd/REPOS/TranskribusBaseLineEvaluationScheme/TranskribusBaseLineEvaluationScheme_v0.1.0/TranskribusBaseLineEvaluationScheme-0.1.0-jar-with-dependencies.jar'
        baseline_evaluator_cmd = ['java', '-jar', baseline_evaluator, '-no_s']

        self.parser = argparse.ArgumentParser(description=parser_description,
                                              fromfile_prefix_chars='@',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                              )
        self.parser.convert_arg_line_to_args = self._convert_file_to_args
        #----------------------------------------------------------------------
        #----- Define general parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--config', default=None, type=str,
                                 help='Use this configuration file')
        self.parser.add_argument('--exp_name', default='layout_exp', type=str, 
                                 help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""")
        self.parser.add_argument('--work_dir', default='./work/', 
                                  type=self._check_out_dir, 
                                  help='Where to place output data')
        self.parser.add_argument('--data_path', default='./data/', 
                                 type=self._check_in_dir, 
                                 help='path to input data')
        self.parser.add_argument('--log_level', default='INFO', type=str,
                                 choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
                                 help='Logging level')
        self.parser.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
                                 type=str, help='Command to evaluate baselines')
        #----------------------------------------------------------------------
        #----- Define preprocessing data parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--img_size', default=[1024,768], nargs=2,
                                 type=self._check_to_int_array, 
                                 help = "Scale images to this size. Format --img_size H W")
        self.parser.add_argument('--line_color', default=128, type=int, 
                                 help='Draw GT lines using this color, range [1,254]')
        self.parser.add_argument('--line_width', default=10, type=int, 
                                 help='Draw GT lines using this number of pixels')
        self.parser.add_argument('--regions', default=regions, nargs='+', 
                                 type=str, 
                                 help="""List of regions to be extracted. 
                                 Format: --regions r1 r2 r3 ...""")
        self.parser.add_argument('--merge_regions', default=None, nargs='+', 
                                 type=str, 
                                 help="""Merge regions on PAGE file into a single one.
                                 Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                                 will be merged into r1 and r5 into r4""")
        #----------------------------------------------------------------------
        #----- Define dataloader parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--batch_size', default=6, type=int,
                                 help='Number of images per mini-batch')
        self.parser.add_argument('--num_workers', default=n_cpus, type=int,
                                 help="""Number of workers used to proces 
                                 input data. If not provided all available
                                 CPUs will be used.
                                 """)
        self.parser.add_argument('--suffle_data', default=True, type=bool,
                                 help='Suffle data during training')
        self.parser.add_argument('--pin_memory', default=True, type=bool,
                                 help='Pin memmory before send to GPU')
        #----------------------------------------------------------------------
        #----- Define NN parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--input_channels', default=3, type=int,
                                 help='Number of channels of input data')
        self.parser.add_argument('--output_channels', default=2, type=int,
                                 help='Number of channels of labels')
        self.parser.add_argument('--cnn_ngf', default=64, type=int,
                                 help='Number of filters of CNNs')
        self.parser.add_argument('--use_gan', default=True, type=bool,
                                 help='USE GAN to compute G loss')
        self.parser.add_argument('--gan_layers', default=3, type=int,
                                 help='Number of layers of GAN NN')
        self.parser.add_argument('--loss_lambda', default=0.001, type=float,
                                help='Lambda weith to copensate GAN vs G loss')
        self.parser.add_argument('--g_loss', default='L1', type=str,
                                 choices=['L1','MSE','smoothL1'],
                                 help='Loss function for G NN')
        #----------------------------------------------------------------------
        #----- Define Optimizer parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--adam_lr', default=0.001, type=float,
                                 help='Initial Lerning rate for ADAM opt')
        self.parser.add_argument('--adam_beta1', default=0.5, type=float,
                                 help='First ADAM exponential decay rate')
        self.parser.add_argument('--adam_beta2', default=0.999, type=float,
                                 help='Secod ADAM exponential decay rate')
        #----------------------------------------------------------------------
        #----- Define Train parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--do_train', default=True, type=bool,
                                 help='Run train stage')
        self.parser.add_argument('--cont_train', default=None, type=str,
                                 help='Continue training using this model')
        self.parser.add_argument('--tr_data', default='./data/train/', type=str,
                                 help="""Train data folder. Train images are
                                 expected there, also PAGE XML files are
                                 expected to be in --tr_data/page folder
                                 """)
        self.parser.add_argument('--epochs', default=100, type=int,
                                 help='Number of training epochs')
        self.parser.add_argument('--tr_img_list', default=None, type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        self.parser.add_argument('--tr_label_list', default=None, type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Test parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--do_test', default=False, type=bool,
                                 help='Run test stage')
        self.parser.add_argument('--te_data', default='./data/test/', type=str,
                                 help="""Test data folder. Test images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """)
        self.parser.add_argument('--te_img_list', default=None, type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        self.parser.add_argument('--te_label_list', default=None, type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Validation parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--do_val', default=False, type=bool,
                                 help='Run Validation stage')
        self.parser.add_argument('--val_data', default='./data/val/', type=str,
                                 help="""Validation data folder. Validation images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """)
        self.parser.add_argument('--val_img_list', default=None, type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        self.parser.add_argument('--val_label_list', default=None, type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Production parameters
        #----------------------------------------------------------------------
        self.parser.add_argument('--do_prod', default=False, type=bool,
                                 help='Run production stage')
        self.parser.add_argument('--prod_data', default='./data/prod/', type=str,
                                 help="""Production data folder. Production images are
                                 expected there.
                                 """)
        self.parser.add_argument('--prod_img_list', default=None, type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        

    def _convert_file_to_args(self,arg_line):
        return arg_line.split(' ')

    def _check_out_dir(self,pointer):
        """ Checks if the dir is wirtable"""
        if (os.path.isdir(pointer)):
            #--- check if is writeable
            if(os.access(pointer,os.W_OK)):
                return pointer
            else:
                raise argparse.ArgumentTypeError('{} folder is not writeable.'.format(pointer))
        else:
            try:
                os.makedirs(pointer)
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError('{} folder does not exist and cannot be created\n{}'.format(e))
    
    def _check_in_dir(self, pointer):
        """check if path exists and is readable"""
        if (os.path.isdir(pointer)):
            if(os.access(pointer, os.R_OK)):
                return pointer
            else:
                raise argparse.ArgumentTypeError('{} folder is not readable.'.format(pointer))
        else:
            raise argparse.ArgumentTypeError('{} folder does not exists'.format(pointer))
    
    def _check_to_int_array(self,data):
        #--- check if size is 2^n compliant
        if ((log(int(data))/log(2)).is_integer()):
            return int(data)
        else:
            raise argparse.ArgumentTypeError('Image size must be 2^n compliant {}'.format(data))

    def _buildColorRegions(self):
        n_class = len(self.opts.regions)
        class_gap = int(256/n_class)
        class_id = class_gap /2
        class_dic = OrderedDict()
        for c in self.opts.regions:
            class_dic[c] = class_id
            class_id = class_id + class_gap

        return class_dic

    def _buildMergedRegions(self):
        if self.opts.merge_regions == None:
            return None
        to_merge = {}
        msg = ''
        for c in self.opts.merge_regions:
            try:
                parent, childs = c.split(':')
                if parent in self.opts.regions:
                    to_merge[parent] = childs.split(',')
                else:
                    msg = '\nRegion "{}" to merge is not defined as region'.format(parent)
                    raise
            except:
                raise argparse.ArgumentTypeError('Malformed argument {}'.format(c) + msg)

        return to_merge

    def parse(self,args=None):
        #--- Parse initialization + command line arguments
        self.opts = self.parser.parse_args(args)
        #--- Parse config file if defined
        if self.opts.config != None:
            logging.info('Reading configuration from {}'.format(self.opts.config))
            self.opts = self.parser.parse_args(['@' + self.opts.config], namespace=self.opts)
        #--- Preprocess some input variables
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir +'/' + self.opts.exp_name + '.log'
        self.opts.regions_colors = self._buildColorRegions()
        self.opts.merged_regions = self._buildMergedRegions()

        return self.opts
    def __str__(self):
        data = '------------ Options -------------'
        for k,v in sorted(vars(self.opts).items()):
            data = data + '\n' + '{0:15}\t{1}'.format(k,v)

        data = data + '\n---------- End  Options ----------\n'
        return data
    def __repr__(self):
        return self.__str__()

        

#------------------------------------------------------------------------------
#-----      Define default values
#------------------------------------------------------------------------------


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ops = arguments()
    data = ops.parse()
    file_logger = logging.FileHandler(data.log_file, mode='w')
    file_logger.setLevel(data.log_level_id)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_logger.setFormatter(formatter)
    #file_logger.FileHandler(data.log_file, mode='w')
    logging.getLogger('').addHandler(file_logger)


    #logging.debug('------------ Options -------------')
    #for k,v in sorted(vars(data).items()):
    #    logging.debug('{0:15}\t{1}'.format(k,v))
    #logging.debug('---------- End  Options ----------')
    logging.debug(ops)

    #print(data)
