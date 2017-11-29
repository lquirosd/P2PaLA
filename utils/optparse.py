from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse 
import os
from math import log
import multiprocessing
import logging

class arguments(object):
    """
    """
    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
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
        general = self.parser.add_argument_group('General Parameters')
        general.add_argument('--config', default=None, type=str,
                                 help='Use this configuration file')
        general.add_argument('--exp_name', default='layout_exp', type=str, 
                                 help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""")
        general.add_argument('--work_dir', default='./work/', 
                                  type=self._check_out_dir, 
                                  help='Where to place output data')
        #--- Removed, input data should be handled by {tr,val,te,prod}_data variables
        #general.add_argument('--data_path', default='./data/', 
        #                         type=self._check_in_dir, 
        #                         help='path to input data')
        general.add_argument('--log_level', default='INFO', type=str,
                                 choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
                                 help='Logging level')
        general.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
                                 type=str, help='Command to evaluate baselines')
        general.add_argument('--num_workers', default=n_cpus, type=int,
                                 help="""Number of workers used to proces 
                                 input data. If not provided all available
                                 CPUs will be used.
                                 """)
        general.add_argument('--gpu', default=0, type=int,
                             help=("GPU id. Use -1 to disable. "
                                   "Only 1 GPU setup is available for now ;("))
        general.add_argument('--no_display', default=False, action='store_true',
                             help='Do not display data on TensorBoard')
        general.add_argument('--use_global_log', default='', type=str,
                             help='Save TensorBoard log on this folder instead default')
        general.add_argument('--log_comment', default='', type=str,
                             help='Add this commaent to TensorBoard logs name')
        #----------------------------------------------------------------------
        #----- Define preprocessing data parameters
        #----------------------------------------------------------------------
        data = self.parser.add_argument_group('Data Related Parameters')
        data.add_argument('--img_size', default=[1024,768], nargs=2,
                                 type=self._check_to_int_array, 
                                 help = "Scale images to this size. Format --img_size H W")
        data.add_argument('--line_color', default=128, type=int, 
                                 help='Draw GT lines using this color, range [1,254]')
        data.add_argument('--line_width', default=10, type=int, 
                                 help='Draw GT lines using this number of pixels')
        data.add_argument('--regions', default=regions, nargs='+', 
                                 type=str, 
                                 help="""List of regions to be extracted. 
                                 Format: --regions r1 r2 r3 ...""")
        data.add_argument('--merge_regions', default=None, nargs='+', 
                                 type=str, 
                                 help="""Merge regions on PAGE file into a single one.
                                 Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                                 will be merged into r1 and r5 into r4""")
        #----------------------------------------------------------------------
        #----- Define dataloader parameters
        #----------------------------------------------------------------------
        loader = self.parser.add_argument_group('Data Loader Parameters')
        loader.add_argument('--batch_size', default=6, type=int,
                                 help='Number of images per mini-batch')
        l_meg1 = loader.add_mutually_exclusive_group(required=False)
        l_meg1.add_argument('--shuffle_data', dest='shuffle_data', action='store_true',
                                 help='Suffle data during training')
        l_meg1.add_argument('--no-shuffle_data', dest='shuffle_data', action='store_false',
                                 help='Do not suffle data during training')
        l_meg1.set_defaults(shuffle_data=True)
        l_meg2 = loader.add_mutually_exclusive_group(required=False)
        l_meg2.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                                 help='Pin memory before send to GPU')
        l_meg2.add_argument('--no-pin_memory', dest='pin_memory', action='store_false',
                                 help='Pin memory before send to GPU')
        l_meg2.set_defaults(pin_memory=True)
        l_meg3 = loader.add_mutually_exclusive_group(required=False)
        l_meg3.add_argument('--flip_img', dest='flip_img', action='store_true',
                                 help='Randomly flip images during training')
        l_meg3.add_argument('--no-flip_img', dest='flip_img', action='store_false',
                                 help='Do not randomly flip images during training')
        l_meg3.set_defaults(flip_img=False)
        #----------------------------------------------------------------------
        #----- Define NN parameters
        #----------------------------------------------------------------------
        net = self.parser.add_argument_group('Neural Networks Parameters')
        net.add_argument('--input_channels', default=3, type=int,
                                 help='Number of channels of input data')
        net.add_argument('--output_channels', default=2, type=int,
                                 help='Number of channels of labels')
        net.add_argument('--cnn_ngf', default=64, type=int,
                                 help='Number of filters of CNNs')
        n_meg = net.add_mutually_exclusive_group(required=False)
        n_meg.add_argument('--use_gan', dest='use_gan', action='store_true',
                                 help='USE GAN to compute G loss')
        n_meg.add_argument('--no-use_gan', dest='use_gan', action='store_false',
                                 help='do not use GAN to compute G loss')
        n_meg.set_defaults(use_gan=True)
        net.add_argument('--gan_layers', default=3, type=int,
                                 help='Number of layers of GAN NN')
        net.add_argument('--loss_lambda', default=0.001, type=float,
                                help='Lambda weith to copensate GAN vs G loss')
        net.add_argument('--g_loss', default='L1', type=str,
                                 choices=['L1','MSE','smoothL1'],
                                 help='Loss function for G NN')
        #----------------------------------------------------------------------
        #----- Define Optimizer parameters
        #----------------------------------------------------------------------
        optim = self.parser.add_argument_group('Optimizer Parameters')
        optim.add_argument('--adam_lr', default=0.001, type=float,
                                 help='Initial Lerning rate for ADAM opt')
        optim.add_argument('--adam_beta1', default=0.5, type=float,
                                 help='First ADAM exponential decay rate')
        optim.add_argument('--adam_beta2', default=0.999, type=float,
                                 help='Secod ADAM exponential decay rate')
        #----------------------------------------------------------------------
        #----- Define Train parameters
        #----------------------------------------------------------------------
        train = self.parser.add_argument_group('Training Parameters')
        tr_meg = train.add_mutually_exclusive_group(required=False)
        tr_meg.add_argument('--do_train', dest='do_train', action='store_true',
                                 help='Run train stage')
        tr_meg.add_argument('--no-do_train', dest='do_train', action='store_false',
                                 help='Do not run train stage')
        tr_meg.set_defaults(do_train=True)
        train.add_argument('--cont_train', default=False, action='store_true',
                                 help='Continue training using this model')
        train.add_argument('--prev_model', default=None, type=str,
                                 help='Use this previously trainned model')
        train.add_argument('--tr_data', default='./data/train/', type=str,
                                 help="""Train data folder. Train images are
                                 expected there, also PAGE XML files are
                                 expected to be in --tr_data/page folder
                                 """)
        train.add_argument('--epochs', default=100, type=int,
                                 help='Number of training epochs')
        train.add_argument('--tr_img_list', default='', type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        train.add_argument('--tr_label_list', default='', type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Test parameters
        #----------------------------------------------------------------------
        test = self.parser.add_argument_group('Test Parameters')
        te_meg = test.add_mutually_exclusive_group(required=False)
        te_meg.add_argument('--do_test', dest='do_test', action='store_true',
                                 help='Run test stage')
        te_meg.add_argument('--no-do_test', dest='do_test', action='store_false',
                                 help='Do not run test stage')
        te_meg.set_defaults(do_test=False)
        test.add_argument('--te_data', default='./data/test/', type=str,
                                 help="""Test data folder. Test images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """)
        test.add_argument('--te_img_list', default='', type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        test.add_argument('--te_label_list', default='', type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Validation parameters
        #----------------------------------------------------------------------
        validation = self.parser.add_argument_group('Validation Parameters')
        v_meg = validation.add_mutually_exclusive_group(required=False)
        v_meg.add_argument('--do_val', dest='do_val', action='store_true',
                                 help='Run Validation stage')
        v_meg.add_argument('--no-do_val', dest='do_val', action='store_false',
                                 help='do not run Validation stage')
        v_meg.set_defaults(do_val=False)
        validation.add_argument('--val_data', default='./data/val/', type=str,
                                 help="""Validation data folder. Validation images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """)
        validation.add_argument('--val_img_list', default='', type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        validation.add_argument('--val_label_list', default='', type=str,
                                 help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        #----------------------------------------------------------------------
        #----- Define Production parameters
        #----------------------------------------------------------------------
        production = self.parser.add_argument_group('Production Parameters')
        p_meg = production.add_mutually_exclusive_group(required=False)
        p_meg.add_argument('--do_prod', dest='do_prod', action='store_true',
                                 help='Run production stage')
        p_meg.add_argument('--no-do_prod', dest='do_prod', action='store_false',
                                 help='Do not run production stage')
        p_meg.set_defaults(do_prod=False)
        production.add_argument('--prod_data', default='./data/prod/', type=str,
                                 help="""Production data folder. Production images are
                                 expected there.
                                 """)
        production.add_argument('--prod_img_list', default='', type=str,
                                 help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """)
        

    def _convert_file_to_args(self,arg_line):
        return arg_line.split(' ')

    def _str_to_bool(self,data):
        """
        Nice way to handle bool flags:
        from: 
        """
        if data.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif data.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def _check_out_dir(self,pointer):
        """ Checks if the dir is wirtable"""
        if (os.path.isdir(pointer)):
            #--- check if is writeable
            if(os.access(pointer,os.W_OK)):
                if not (os.path.isdir(pointer + '/checkpoints')):
                    os.makedirs(pointer + '/checkpoints')
                    self.logger.debug('Creating checkpoints dir: {}'.format(pointer + '/checkpoints'))
                return pointer
            else:
                raise argparse.ArgumentTypeError('{} folder is not writeable.'.format(pointer))
        else:
            try:
                os.makedirs(pointer)
                self.logger.debug('Creating output dir: {}'.format(pointer))
                os.makedirs(pointer + '/checkpoints')
                self.logger.debug('Creating checkpoints dir: {}'.format(pointer + '/checkpoints'))
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
        data = int(data)
        if (data > 0 and data%256 == 0):
            return data
        else:
            raise argparse.ArgumentTypeError('Image size must be multiple of 256: {} is not'.format(data))

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

    def parse(self):
        #--- Parse initialization + command line arguments
        self.opts = self.parser.parse_args()
        #--- Parse config file if defined
        if self.opts.config != None:
            self.logger.info('Reading configuration from {}'.format(self.opts.config))
            self.opts = self.parser.parse_args(['@' + self.opts.config], namespace=self.opts)
            self.opts = self.parser.parse_args(namespace=self.opts)
        #--- Preprocess some input variables
        self.opts.use_gpu = self.opts.gpu != -1
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir +'/' + self.opts.exp_name + '.log'
        self.opts.regions_colors = self._buildColorRegions()
        self.opts.merged_regions = self._buildMergedRegions()
        self.opts.checkpoints = os.path.join(self.opts.work_dir, 'checkpoints/')

        return self.opts
    def __str__(self):
        data = '------------ Options -------------'
        try:
            for k,v in sorted(vars(self.opts).items()):
                data = data + '\n' + '{0:15}\t{1}'.format(k,v)
        except:
            data = data + '\nNo arguments parsed yet...'
         
        data = data + '\n---------- End  Options ----------\n'
        return data
    def __repr__(self):
        return self.__str__()

