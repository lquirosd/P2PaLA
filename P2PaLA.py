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
import signal
#import gc

import torch
#from torchvision import  utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.optparse import Arguments as arguments
from nn_models import models
from data import dataset
from data import transforms as transforms
from data import imgprocess as dp
import cPickle as pickle
from evalTools import  page2page_eval

loss_dic = {'L1':torch.nn.L1Loss(size_average=True),
            'MSE':torch.nn.MSELoss(size_average=True),
            'smoothL1':torch.nn.SmoothL1Loss(size_average=True),
            'NLL':torch.nn.NLLLoss(size_average=True)}

def tensor2img(image_tensor, imtype=np.uint8):
    #--- function just for debug, do not use on production stage
    #-- @@@@@@
    i_dim = image_tensor.size()
    ex_dim = torch.ones(3-i_dim[0],i_dim[1],i_dim[2])
    ex_dim = ex_dim.cuda()
    v_o = torch.cat((image_tensor,ex_dim),dim=0)
    image_numpy = v_o.cpu().float().numpy()
    image_numpy = ((np.transpose(image_numpy, (1, 2, 0)))+1) * 127.5
    return image_numpy.astype(imtype)

#def signal_handler(signal,frame):
#    stop_current_job = True

def save_checkpoint(state, is_best, opts, logger, epoch, criterion=''):
    """
    Save current model to checkpoints dir
    """
    #--- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    if is_best:
        out_file = os.path.join(opts.checkpoints, "".join(['best_under',criterion,'criterion.pth']))
        torch.save(state, out_file)
        logger.info('Best model saved to {} at epoch {}'.format(out_file, str(epoch)))
    else:
        out_file = os.path.join(opts.checkpoints, 'checkpoint.pth')
        torch.save(state, out_file)
        logger.info('Checkpoint saved to {} at epoch {}'.format(out_file, str(epoch)))
    return out_file

def check_inputs(opts, logger):
    """
    check if some inputs are correct
    """
    n_err = 0
    #--- check if input files/folders exists
    if opts.do_train:
        if opts.tr_img_list == '':
            if not (os.path.isdir(opts.tr_data) and os.access(opts.tr_data, os.R_OK)):
                n_err = n_err + 1
                logger.error('Folder {} does not exists or is unreadable'.format(opts.tr_data))
        else:
            if not (os.path.isfile(opts.tr_img_list) and os.access(opts.tr_img_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.tr_img_list))
            if not (os.path.isfile(opts.tr_label_list) and os.access(opts.tr_label_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.tr_label_list))
    if opts.do_test:
        if opts.te_img_list == '':
            if not (os.path.isdir(opts.te_data) and os.access(opts.te_data, os.R_OK)):
                n_err = n_err + 1
                logger.error('Folder {} does not exists or is unreadable'.format(opts.te_data))
        else:
            if not (os.path.isfile(opts.te_img_list) and os.access(opts.te_img_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.te_img_list))
            if not (os.path.isfile(opts.te_label_list) and os.access(opts.te_label_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.te_label_list))
    if opts.do_val:
        if opts.val_img_list == '':
            if not (os.path.isdir(opts.val_data) and os.access(opts.val_data, os.R_OK)):
                n_err = n_err + 1
                logger.error('Folder {} does not exists or is unreadable'.format(opts.val_data))
        else:
            if not (os.path.isfile(opts.val_img_list) and os.access(opts.val_img_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.val_img_list))
            if not (os.path.isfile(opts.val_label_list) and os.access(opts.val_label_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.val_label_list))
    if opts.do_prod:
        if opts.prod_img_list == '':
            if not (os.path.isdir(opts.prod_data) and os.access(opts.prod_data, os.R_OK)):
                n_err = n_err + 1
                logger.error('Folder {} does not exists or is unreadable'.format(opts.prod_data))
        else:
            if not (os.path.isfile(opts.prod_img_list) and os.access(opts.prod_img_list, os.R_OK)):
                n_err = n_err + 1
                logger.error('File {} does not exists or is unreadable'.format(opts.prod_img_list))
    #--- if cont_train is defined prev_model must be defined as well
    if opts.cont_train:
        if opts.prev_model == None:
            n_err = n_err + 1
            logger.error('--prev_model must be defined to perform continue training.')
        else:
            if not (os.path.isfile(opts.prev_model) and os.access(opts.prev_model, os.R_OK)):
                n_err = n_err + 1 
                logger.error('File {} does not exists or is unreadable'.format(opts.prev_model))
        if not opts.do_train:
            logger.warning(("Continue training is defined, but train stage is not. "
                           "Skipping continue..."))
    #--- if test,val or prod is performed, train or prev model must be defined
    if opts.do_val:
        if not opts.do_train:
            logger.warning(("Validation data runs only under train stage, but " 
                            "no train stage is running. Skipping validation ..."))
    if opts.do_test or opts.do_prod:
        if not (opts.do_train):
            if opts.prev_model == None:
                n_err = n_err + 1
                logger.error(("There is no model available through training or "
                              "previously trained model. "
                              "Test and Production stages cannot be performed..."))
            else:
                if not (os.path.isfile(opts.prev_model) and os.access(opts.prev_model, os.R_OK)):
                    n_err = n_err + 1 
                    logger.error('File {} does not exists or is unreadable'.format(opts.prev_model))

    return n_err 


def main():
    """
    """
    global_start = time.time()
    #--- init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    #--- keep this logger at DEBUG level, until aguments are processed 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #--- handle Ctrl-C signal
    #signal.signal(signal.SIGINT,signal_handler)

    #--- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    if check_inputs(opts,logger):
        logger.critical('Execution aborted due input errors...')
        exit(1)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(opts.log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #--- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    logger.debug(in_args)
    #--- Init torch random 
    #--- This two are suposed to be merged in the future, for now keep boot
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    #--- Init model variable
    nnG = None
    bestState = None
    prior=None
    torch.set_default_tensor_type("torch.FloatTensor")
    #--- configure TensorBoard display
    opts.img_size = np.array(opts.img_size, dtype=np.int)
    #--------------------------------------------------------------------------
    #-----  TRAIN STEP
    #--------------------------------------------------------------------------
    if opts.do_train:
        train_start = time.time()
        logger.info('Working on training stage...')
        #--- display is used only on training step
        if not opts.no_display:
            import socket
            from datetime import datetime
            try:
                from tensorboardX import SummaryWriter
                if opts.use_global_log:
                    run_dir = opts.use_global_log
                else:
                    run_dir = os.path.join(opts.work_dir, 'runs')
                log_dir = os.path.join(run_dir, 
                               "".join([datetime.now().strftime('%b%d_%H-%M-%S'),
                               '_',socket.gethostname(), opts.log_comment]))
                                    
                writer = SummaryWriter(log_dir=log_dir) 
                logger.info('TensorBoard log will be stored at {}'.format(log_dir))
                logger.info('run: tensorboard --logdir {}'.format(run_dir))
            except:
                logger.warning('tensorboardX is not installed, display logger set to OFF.')
                opts.no_display = True
    
        #--- Build transforms
        transform = transforms.build_transforms(opts,train=True)
        #--- Get Train Data
        if opts.tr_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.tr_data))
            tr_data = dp.htrDataProcess(
                                         opts.tr_data,
                                         os.path.join(opts.work_dir,'data','train'),
                                         opts,
                                         logger=logger)
            tr_data.pre_process()
            opts.tr_img_list = tr_data.img_list
            opts.tr_label_list = tr_data.label_list
        else:
            logger.info('Reading data from pre-processed input {}'.format(opts.tr_img_list))
            tr_data = dp.htrDataProcess(
                                        opts.tr_data,
                                        os.path.join(opts.work_dir,'data','train'),
                                        opts,
                                        logger=logger)
            tr_data.set_img_list(opts.tr_img_list)
            tr_data.set_label_list(opts.tr_label_list)

        train_data = dataset.htrDataset(img_lst=opts.tr_img_list,
                                        label_lst=opts.tr_label_list,
                                        transform=transform,
                                        opts=opts)
        if opts.do_prior:
            #--- Save prior matrix along with model
            fh = open(os.path.join(opts.checkpoints, 'prior.pth'),'w')
            pickle.dump(train_data.prior,fh,-1)
            fh.close()
        train_dataloader = DataLoader(train_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)
        #--- Get Val data, if needed
        if opts.do_val:
            if opts.val_img_list == '':
                logger.info('Preprocessing data from {}'.format(opts.val_data))
                va_data = dp.htrDataProcess(
                                             opts.val_data,
                                             os.path.join(opts.work_dir,'data','val/'),
                                             opts,
                                             logger=logger)
                va_data.pre_process()
                opts.val_img_list = va_data.img_list
                opts.val_label_list = va_data.label_list
            val_transform = transforms.build_transforms(opts,train=False)

            val_data = dataset.htrDataset(img_lst=opts.val_img_list,
                                          label_lst=opts.val_label_list,
                                          transform=val_transform,
                                          opts=opts)
            val_dataloader = DataLoader(val_data,
                                        batch_size=opts.batch_size,
                                        shuffle=False,
                                        num_workers=opts.num_workers,
                                        pin_memory=opts.pin_memory)

        #--- Build Models
        nnG = models.buildUnet(opts.input_channels,
                               opts.output_channels,
                               ngf=opts.cnn_ngf,
                               net_type=opts.net_out_type,
                               out_mode=opts.out_mode)
        #--- TODO: create a funtion @ models to define loss function
        #--- TODO: create a funtion @ models to define loss function
        if opts.do_class:
            lossG = loss_dic['NLL']
            opts.g_loss = 'NLL'
        else:
            lossG = loss_dic[opts.g_loss]
        #--- TODO: implement other initializadion methods
        optimizerG = optim.Adam(nnG.parameters(),
                                lr=opts.adam_lr,
                                betas=(opts.adam_beta1,opts.adam_beta2))
        if opts.cont_train:
            logger.info('Resumming training from model {}'.format(opts.prev_model))
            checkpoint = torch.load(opts.prev_model)
            nnG.load_state_dict(checkpoint['nnG_state'])
            optimizerG.load_state_dict(checkpoint['nnG_optimizer_state'])
            if not opts.g_loss == checkpoint['g_loss']:
                logger.warning(("Previous Model loss function differs from "
                                "current loss funtion {} != {}").format(
                                                                opts.g_loss,
                                                                checkpoint['g_loss']))
                logger.warning('Using {} loss funtion to resume training...'.format(opts.g_loss))
            if opts.use_gpu:
                nnG = nnG.cuda()
                lossG = lossG.cuda()
        else:
            #--- send to GPU before init weigths
            if opts.use_gpu:
                nnG = nnG.cuda()
                lossG = lossG.cuda()
            nnG.apply(models.weights_init_normal)
        logger.debug('GEN Network:\n{}'.format(nnG)) 
        logger.debug('GEN Network, number of parameters: {}'.format(nnG.num_params))

        if opts.use_gan:
            if opts.net_out_type == 'C':
                if opts.out_mode == 'LR':
                    d_out_ch = 2
                else:
                    d_out_ch = 1
            elif opts.net_out_type == 'R':
                d_out_ch = opts.output_channels
            else:
                pass
            nnD = models.buildDNet(opts.input_channels,
                                   d_out_ch,
                                   ngf=opts.cnn_ngf,
                                   n_layers=opts.gan_layers)
            lossD = torch.nn.BCELoss(size_average=True)
            optimizerD = optim.Adam(nnD.parameters(),
                                    lr=opts.adam_lr,
                                    betas=(opts.adam_beta1,opts.adam_beta2))
            if opts.cont_train:
                if 'nnD_state' in checkpoint:
                    nnD.load_state_dict(checkpoint['nnD_state'])
                    optimizerD.load_state_dict(checkpoint['nnD_optimizer_state'])
                else:
                    logger.warning('Previous model did not use GAN, but current does.')
                    logger.warning('Using new GAN model from scratch.')
                if opts.use_gpu:
                    nnD = nnD.cuda()
                    lossD = lossD.cuda()
                    #loss_lambda = loss_lambda.cuda()
            else:
                if opts.use_gpu:
                    nnD = nnD.cuda()
                    lossD = lossD.cuda()
                    #loss_lambda = loss_lambda.cuda()
                nnD.apply(models.weights_init_normal) 
            logger.debug('DIS Network:\n{}'.format(nnD)) 
            logger.debug('DIS Network, number of parameters: {}'.format(nnD.num_params))

        #--- Do the actual train
        #--- TODO: compute statistical boostrap to define if a model is
        #---    statistically better than previous
        best_val = np.inf
        best_tr = np.inf
        best_model = ''
        best_epoch = 0
        if opts.net_out_type == 'C' and opts.fix_class_imbalance:
            if opts.out_mode == 'LR':
                l_w = torch.from_numpy(train_data.w[0])
                r_w = torch.from_numpy(train_data.w[1])
                if opts.use_gpu:
                    l_w = l_w.type(torch.FloatTensor).cuda()
                    r_w = r_w.type(torch.FloatTensor).cuda()
                class_weight = [l_w,r_w]
                logger.debug('class weight: {}'.format(train_data.w))
            else:
                lossG.weight = torch.from_numpy(train_data.w).type(torch.FloatTensor).cuda()
                logger.debug('class weight: {}'.format(train_data.w))

        for epoch in range(opts.epochs):
            epoch_start = time.time()
            epoch_lossG = 0
            epoch_lossGAN = 0
            epoch_lossR = 0
            epoch_lossD = 0
            for batch,sample in enumerate(train_dataloader):
                #--- Reset Grads
                #nnG.apply(models.zero_bias)
                optimizerG.zero_grad()
                x = Variable(sample['image'], requires_grad=False)
                #y_gt_D = Variable(sample['label'].clone().type(torch.FloatTensor), requires_grad=False)
                y_gt = Variable(sample['label'], requires_grad=False)
                if opts.use_gpu:
                    x = x.cuda()
                    y_gt = y_gt.cuda()
                    #y_gt_D = y_gt_D.cuda()
                y_gen = nnG(x)
                if opts.out_mode == 'LR' and opts.net_out_type == 'C':
                    if (y_gen[0] != y_gen[0]).any() or (y_gen[1] != y_gen[1]).any():
                        logger.error('NaN values found in hypotesis')
                        logger.error("Inputs: {}".format(sample['id']))
                        raise RuntimeError 
                    y_l,y_r = torch.split(y_gt,1,dim=1)
                    if opts.fix_class_imbalance:
                        lossG.weight = class_weight[0]
                        g_loss = lossG(y_gen[0],torch.squeeze(y_l))
                        lossG.weight = class_weight[1]
                        g_loss += lossG(y_gen[1],torch.squeeze(y_r))
                    else:
                        g_loss = lossG(y_gen[0],torch.squeeze(y_l)) + lossG(y_gen[1],torch.squeeze(y_r))
                    #g_loss = lossG(y_gen[0],torch.squeeze(y_l)) + lossG(y_gen[1],torch.squeeze(y_r))
                else:
                    if (y_gen != y_gen).any():
                        logger.error('NaN values found in hypotesis')
                        logger.error("Inputs: {}".format(sample['id']))
                        raise RuntimeError 
                    g_loss = lossG(y_gen,y_gt)
                #--- reduce is not implemented, average is implemented in loss
                #--- function itself
                #g_loss = g_loss * (1/y_gen.data[0].numel())
                if opts.use_gan:
                    #nnD.apply(models.zero_bias)
                    optimizerD.zero_grad()
                    if opts.net_out_type == 'C':
                        if opts.out_mode == 'LR':
                            real_D = torch.cat([x,y_gt.type(torch.cuda.FloatTensor)],1)
                            #real_D = torch.cat([x,y_gt_D],1)
                            y_dis_real = nnD(real_D)
                            _, arg_l = torch.max(y_gen[0],dim=1,keepdim=True)
                            _, arg_r = torch.max(y_gen[1],dim=1,keepdim=True)
                            y_fake = torch.cat([arg_l,arg_r],1)
                            fake_D = torch.cat([x,y_fake.type(torch.cuda.FloatTensor)],1).detach()
                        elif opts.out_mode == 'L' or opts.out_mode == 'R':
                            real_D = torch.cat([x,torch.unsqueeze(y_gt.type(torch.cuda.FloatTensor),1)],1)
                            y_dis_real = nnD(real_D)
                            _, arg_y = torch.max(y_gen,dim=1)
                            fake_D = torch.cat([x,torch.unsqueeze(arg_y.type(torch.cuda.FloatTensor),1)],1).detach()
                        else:
                            pass
                    else:
                        real_D = torch.cat([x,y_gt.type(torch.cuda.FloatTensor)],1)
                        y_dis_real = nnD(real_D)
                        fake_D = torch.cat([x,y_gen],1).detach()
                    y_dis_fake = nnD(fake_D) 
                    label_D_size = y_dis_real.size()
                    real_y = Variable(torch.FloatTensor(label_D_size).fill_(1.0),
                                      requires_grad=False)
                    fake_y = Variable(torch.FloatTensor(label_D_size).fill_(0.0),
                                      requires_grad=False)
                    if opts.use_gpu:
                        real_y = real_y.cuda()
                        fake_y = fake_y.cuda()
                    d_loss_real = lossD(y_dis_real,real_y)
                    d_loss_fake = lossD(y_dis_fake,fake_y)
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    epoch_lossD += d_loss.data[0]
                    d_loss.backward()
                    optimizerD.step()
                    if opts.net_out_type == 'C':
                        if opts.out_mode == 'LR':
                            _, arg_l = torch.max(y_gen[0],dim=1,keepdim=True)
                            _,arg_r = torch.max(y_gen[1],dim=1,keepdim=True)
                            y_fake = torch.cat([arg_l,arg_r],1)
                            g_fake = torch.cat([x,y_fake.type(torch.cuda.FloatTensor)],1)
                        elif opts.out_mode == 'L' or opts.out_mode == 'R':
                            _, arg_y = torch.max(y_gen,dim=1,keepdim=True)
                            g_fake = torch.cat([x,arg_y.type(torch.cuda.FloatTensor)],1)
                        else:
                            pass
                    else:
                        g_fake = torch.cat([x,y_gen],1)
                    g_y = nnD(g_fake)
                    shared_loss = lossD(g_y,real_y) 
                    epoch_lossR += shared_loss.data[0]
                    gan_loss = (shared_loss + (g_loss * opts.loss_lambda))
                else:
                    gan_loss = g_loss
                epoch_lossG += g_loss.data[0] / y_gt.data.size()[0]
                epoch_lossGAN += gan_loss.data[0] / y_gt.data.size()[0]
                gan_loss.backward()
                optimizerG.step()
            #--- forward pass val
            if opts.do_val:
                val_loss = 0
                for v_batch,v_sample in enumerate(val_dataloader):
                    #--- set vars to volatile, since bo backward used
                    v_img = Variable(v_sample['image'], volatile=True)
                    v_label = Variable(v_sample['label'], volatile=True)
                    if opts.use_gpu:
                        v_img = v_img.cuda()
                        v_label = v_label.cuda()
                    v_y = nnG(v_img)
                    if opts.out_mode == 'LR' and opts.net_out_type == 'C':
                        v_l,v_r = torch.split(v_label,1,dim=1)
                        lossG.weight = None
                        v_loss = lossG(v_y[0],torch.squeeze(v_l)) + lossG(v_y[1],torch.squeeze(v_r))
                    else:
                        v_loss = lossG(v_y, v_label)
                    #v_loss = v_loss * (1/v_y.data[0].numel())
                    val_loss += v_loss.data[0] / v_label.data.size()[0]
                val_loss = val_loss/v_batch
            #--- Write to Logs
            if not opts.no_display:
                writer.add_scalar('train/lossGAN',epoch_lossGAN/batch,epoch)
                writer.add_scalar('train/lossG',epoch_lossG/batch,epoch)
                writer.add_text('LOG', 'End of epoch {0} of {1} time Taken: {2:.3f} sec'.format(
                             str(epoch),str(opts.epochs),
                             time.time()-epoch_start), epoch)
                if opts.use_gan:
                    writer.add_scalar('train/lossD',epoch_lossD/batch,epoch)
                    writer.add_scalar('train/D_loss_Real',epoch_lossR/batch,epoch)
                if opts.do_val:
                    writer.add_scalar('val/lossG',val_loss,epoch)
            #--- Save model under val or min loss
            if opts.do_val:
                if best_val >= val_loss:
                    best_epoch = epoch
                    state = {
                            'nnG_state':            nnG.state_dict(),
                            'nnG_optimizer_state':  optimizerG.state_dict(),
                            'g_loss':               opts.g_loss
                            }
                    if opts.use_gan:
                        state['nnD_state'] =            nnD.state_dict()
                        state['nnD_optimizer_state'] =  optimizerD.state_dict()
                    best_model = save_checkpoint(state, True, opts, logger, epoch,
                                                 criterion='val' + opts.g_loss)
                    logger.info("New best model, from {} to {}".format(best_val,val_loss))
                    best_val = val_loss
            else:
                if best_tr >= epoch_lossG:
                    best_epoch = epoch
                    state = {
                            'nnG_state':            nnG.state_dict(),
                            'nnG_optimizer_state':  optimizerG.state_dict(),
                            'g_loss':               opts.g_loss
                            }
                    if opts.use_gan:
                        state['nnD_state'] =            nnD.state_dict()
                        state['nnD_optimizer_state'] =  optimizerD.state_dict()
                    best_model = save_checkpoint(state, True, opts, logger, epoch,
                                                 criterion=opts.g_loss)
                    logger.info("New best model, from {} to {}".format(best_tr,epoch_lossG))
                    best_tr = epoch_lossG
            #--- Save checkpoint
            if epoch%opts.save_rate == 0 or epoch == opts.epochs - 1:
                #--- save current model, to test load func
                state = {
                        'nnG_state':            nnG.state_dict(),
                        'nnG_optimizer_state':  optimizerG.state_dict(),
                        'g_loss':               opts.g_loss
                        }
                if opts.use_gan:
                    state['nnD_state'] =            nnD.state_dict()
                    state['nnD_optimizer_state'] =  optimizerD.state_dict()
                best_model = save_checkpoint(state, False, opts, logger, epoch)
        
        logger.info('Trining stage done. total time taken: {}'.format(time.time()-train_start))
        #---- Train is done, next is to save validation inference
        if opts.do_val:
            logger.info('Working on validation inference...')
            res_path = os.path.join(opts.work_dir, 'results', 'val')
            try:
                os.makedirs(os.path.join(res_path,'page'))
                os.makedirs(os.path.join(res_path,'mask'))
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(
                                    os.path.join(res_path,'page')):
                    pass
                else:
                    raise
            if opts.save_prob_mat:
                try:
                    os.makedirs(os.path.join(res_path,'prob_mat'))
                except OSError as exc:
                    if exc.errno == errno.EEXIST and os.path.isdir(res_path + '/prob_mat'):
                        pass
                    else:
                        raise
            #--- Set model to eval, to perform inference step 
            if best_epoch == epoch:
                nnG.eval()
                if opts.do_off:
                    nnG.apply(models.off_dropout)
            else:
                #--- load best model for inference
                checkpoint = torch.load(best_model)
                nnG.load_state_dict(checkpoint['nnG_state'])
                if opts.use_gpu:
                    nnG = nnG.cuda()
                nnG.eval()
                if opts.do_off:
                    nnG.apply(models.off_dropout)

            #--- get prior data
            if opts.do_prior and prior==None:
                fh = open(os.path.join(opts.checkpoints, 'prior.pth'),'r')
                prior = pickle.load(fh)
                fh.close()
                if opts.out_mode == 'LR':
                    priorL = Variable(torch.from_numpy(np.log(prior[0])).type(torch.FloatTensor))
                    #print(priorL.shape)
                    priorL = priorL.cuda()
                    priorR = Variable(torch.from_numpy(np.log(prior[1])).type(torch.FloatTensor))
                    priorR = priorR.cuda()
                elif opts.out_mode == 'L' or opts.out_mode == 'R':
                    prior = Variable(torch.from_numpy(np.log(prior)).type(torch.FloatTensor))
                    prior = prior.cuda()
            for v_batch,v_sample in enumerate(val_dataloader):
                #--- set vars to volatile, since no backward used
                v_img = Variable(v_sample['image'], volatile=True)
                v_label = Variable(v_sample['label'], volatile=True)
                v_ids = v_sample['id']
                if opts.use_gpu:
                    v_img = v_img.cuda()
                    v_label = v_label.cuda()
                v_y_gen = nnG(v_img)
                if opts.save_prob_mat:
                    for idx,data in enumerate(v_y_gen.data):
                        fh = open(res_path + '/prob_mat/' + v_ids[idx] + '.pickle', 'w')
                        pickle.dump(data.cpu().float().numpy(),fh,-1)
                        fh.close
                if opts.net_out_type == 'C':
                    if opts.out_mode == 'LR':
                        if opts.do_prior:
                            v_y_gen[0].data = v_y_gen[0].data+priorL.data
                            v_y_gen[1].data = v_y_gen[1].data+priorR.data
                        _, v_l = torch.max(v_y_gen[0],dim=1,keepdim=True)
                        _, v_r = torch.max(v_y_gen[1],dim=1,keepdim=True)
                        v_y_gen = torch.cat([v_l, v_r],1)
                    elif opts.out_mode == 'L' or opts.out_mode == 'R':
                        if opts.do_prior:
                            v_y_gen.data = v_y_gen.data + prior.data
                        _, v_y_gen = torch.max(v_y_gen,dim=1,keepdim=True)
                    else:
                        pass
                elif opts.net_out_type == 'R':
                    pass
                else:
                    pass
                #--- save out as image for visual check
                #for idx,data in enumerate(v_label.data):
                #    img = tensor2img(data)
                #    cv2.imwrite(os.path.join(res_path,
                #                             'mask', v_ids[idx] +'_gt.png'),img)
                for idx,data in enumerate(v_y_gen.data):
                    #img = tensor2img(data)
                    #cv2.imwrite(os.path.join(res_path,
                    #                         'mask', v_ids[idx] +'_out.png'),img)
                    va_data.gen_page(v_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)
            #--- metrics are taked over the generated PAGE-XML files instead
            #--- of teh current data and label becouse image size may be different
            #--- than the processed image, then during evaluation final image
            #--- must be used
            va_results = page2page_eval.compute_metrics(va_data.hyp_xml_list,
                                                        va_data.gt_xml_list,
                                                        opts)
            logger.info('-'*10 + 'VALIDARION RESULTS SUMMARY' + '-'*10)
            logger.info(','.join(va_results.keys()))
            logger.info(','.join(str(x) for x in va_results.values()))
        if not opts.no_display:
            writer.close()
    
    #--------------------------------------------------------------------------
    #---    TEST INFERENCE
    #--------------------------------------------------------------------------
    if opts.do_test:
        logger.info('Working on test inference...')
        res_path = os.path.join(opts.work_dir, 'results', 'test')
        try:
            os.makedirs(os.path.join(res_path,'page'))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(res_path + '/page'):
                pass
            else:
                raise
        if opts.save_prob_mat:
            try:
                os.makedirs(os.path.join(res_path,'prob_mat'))
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(res_path + '/prob_mat'):
                    pass
                else:
                    raise
        logger.info('Results will be saved to {}'.format(res_path))

        if nnG == None:
            #--- Load Model 
            nnG = models.buildUnet(opts.input_channels,
                                   opts.output_channels,
                                   ngf=opts.cnn_ngf,
                                   net_type=opts.net_out_type,
                                   out_mode=opts.out_mode)
            logger.info('Resumming from model {}'.format(opts.prev_model))
            checkpoint = torch.load(opts.prev_model)
            nnG.load_state_dict(checkpoint['nnG_state'])
            if opts.use_gpu:
                nnG = nnG.cuda()
            nnG.eval()
            if opts.do_off:
                nnG.apply(models.off_dropout)
            logger.debug('GEN Network:\n{}'.format(nnG)) 
            logger.debug('GEN Network, number of parameters: {}'.format(nnG.num_params))
        else:
            logger.debug('Using prevously loaded Generative module for test...')
            nnG.eval()
            if opts.do_off:
                nnG.apply(models.off_dropout)

        #--- get test data
        test_start_time = time.time()
        if opts.te_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.te_data))
            te_data = dp.htrDataProcess(
                                         opts.te_data,
                                         os.path.join(opts.work_dir,'data','test'),
                                         opts,
                                         logger=logger)
            te_data.pre_process()
            opts.te_img_list = te_data.img_list
            opts.te_label_list = te_data.label_list
        
        transform = transforms.build_transforms(opts,train=False)

        test_data = dataset.htrDataset(img_lst=opts.te_img_list,
                                        label_lst=opts.te_label_list,
                                        transform=transform,
                                        opts=opts)
        test_dataloader = DataLoader(test_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)
        #--- get prior data
        if opts.do_prior and prior==None:
            fh = open(os.path.join(opts.checkpoints, 'prior.pth'),'r')
            prior = pickle.load(fh)
            fh.close()
            if opts.out_mode == 'LR':
                priorL = Variable(torch.from_numpy(np.log(prior[0])).type(torch.FloatTensor))
                #print(priorL.shape)
                priorL = priorL.cuda()
                priorR = Variable(torch.from_numpy(np.log(prior[1])).type(torch.FloatTensor))
                priorR = priorR.cuda()
            elif opts.out_mode == 'L' or opts.out_mode == 'R':
                prior = Variable(torch.from_numpy(np.log(prior)).type(torch.FloatTensor))
                prior = prior.cuda()
        for te_batch,sample in enumerate(test_dataloader):
            te_x = Variable(sample['image'], volatile=True)
            te_label = Variable(sample['label'], volatile=True)
            te_ids = sample['id']
            if opts.use_gpu:
                te_x = te_x.cuda()
                te_label = te_label.cuda()
            te_y_gen = nnG(te_x)
            if opts.save_prob_mat:
                if opts.out_mode == 'LR':
                    for idx,data in enumerate(te_y_gen[0].data):
                        fh = open(res_path + '/prob_mat/' + te_ids[idx] + '.pickle', 'w')
                        pickle.dump(tuple((data.cpu().float().numpy(),
                            te_y_gen[1].data.cpu().float().numpy()
                            )),fh,-1)
                        fh.close()
                else:
                    for idx,data in enumerate(te_y_gen.data):
                        fh = open(res_path + '/prob_mat/' + te_ids[idx] + '.pickle', 'w')
                        pickle.dump(data.cpu().float().numpy(),fh,-1)
                        fh.close
            if opts.net_out_type == 'C':
                if opts.out_mode == 'LR':
                    #print(te_y_gen[0].data.shape)
                    if opts.do_prior:
                        te_y_gen[0].data = te_y_gen[0].data+priorL.data
                        te_y_gen[1].data = te_y_gen[1].data+priorR.data
                    _, te_l = torch.max(te_y_gen[0],dim=1,keepdim=True)
                    _, te_r = torch.max(te_y_gen[1],dim=1,keepdim=True)
                    te_y_gen = torch.cat([te_l, te_r],1)
                elif opts.out_mode == 'L' or opts.out_mode == 'R':
                    if opts.do_prior:
                        te_y_gen.data = te_y_gen.data + prior.data
                    _, te_y_gen = torch.max(te_y_gen,dim=1,keepdim=True)
                else:
                    pass
            elif opts.net_out_type == 'R':
                pass
            else:
                pass

            for idx,data in enumerate(te_y_gen.data):
                #--- TODO: update this function to proccess C-dim tensors
                te_data.gen_page(te_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)
        test_end_time = time.time()
        logger.info('Test stage done. total time taken: {}'.format(test_end_time-test_start_time))
        logger.info('Average time per page: {}'.format((test_end_time-test_start_time)/test_data.__len__()))
        #--- metrics are taked over the generated PAGE-XML files instead
        #--- of teh current data and label becouse image size may be different
        #--- than the processed image, then during evaluation final image
        #--- must be used
        te_results = page2page_eval.compute_metrics(te_data.hyp_xml_list,
                                                    te_data.gt_xml_list,
                                                    opts, logger=logger) 
        logger.info('-'*10 + 'TEST RESULTS SUMMARY' + '-'*10)
        logger.info(','.join(te_results.keys()))
        logger.info(','.join(str(x) for x in te_results.values()))
    #--------------------------------------------------------------------------
    #---    PRODUCTION INFERENCE
    #--------------------------------------------------------------------------
    if opts.do_prod:
        logger.info('Working on prod inference...')
        res_path = os.path.join(opts.work_dir, 'results', 'prod')
        try:
            os.makedirs(os.path.join(res_path,'page'))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(res_path + '/page'):
                pass
            else:
                raise
        if opts.save_prob_mat:
            try:
                os.makedirs(os.path.join(res_path,'prob_mat'))
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(res_path + '/prob_mat'):
                    pass
                else:
                    raise
        logger.info('Results will be saved to {}'.format(res_path))

        if nnG == None:
            #--- Load Model 
            nnG = models.buildUnet(opts.input_channels,
                                   opts.output_channels,
                                   ngf=opts.cnn_ngf,
                                   net_type=opts.net_out_type,
                                   out_mode=opts.out_mode)
            logger.info('Resumming from model {}'.format(opts.prev_model))
            checkpoint = torch.load(opts.prev_model)
            nnG.load_state_dict(checkpoint['nnG_state'])
            if opts.use_gpu:
                nnG = nnG.cuda()
            nnG.eval()
            if opts.do_off:
                nnG.apply(models.off_dropout)
            logger.debug('GEN Network:\n{}'.format(nnG)) 
            logger.debug('GEN Network, number of parameters: {}'.format(nnG.num_params))
        else:
            logger.debug('Using prevously loaded Generative module for prod...')
            nnG.eval()
            if opts.do_off:
                nnG.apply(models.off_dropout)

        #--- get prod data
        prod_start_time = time.time()
        pr_data = dp.htrDataProcess(
                                    opts.prod_data,
                                    os.path.join(opts.work_dir,'data','prod'),
                                    opts,
                                    build_labels=False,
                                    logger=logger)
        if opts.prod_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.prod_data))
            #pr_data = dp.htrDataProcess(
            #                             opts.prod_data,
            #                             os.path.join(opts.work_dir,'data','prod'),
            #                             opts,
            #                             build_labels=False,
            #                             logger=logger)
            pr_data.pre_process()
            opts.prod_img_list = pr_data.img_list
        else:
            logger.info('Loading pre-processed data from {}'.format(opts.prod_img_list))
            pr_data.set_img_list(opts.prod_img_list)

        
        transform = transforms.build_transforms(opts,train=False)

        prod_data = dataset.htrDataset(img_lst=opts.prod_img_list,
                                       transform=transform,
                                       opts=opts)
        prod_dataloader = DataLoader(prod_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)

        #--- get prior data
        if opts.do_prior and prior==None:
            fh = open(os.path.join(opts.checkpoints, 'prior.pth'),'r')
            prior = pickle.load(fh)
            fh.close()
            if opts.out_mode == 'LR':
                priorL = Variable(torch.from_numpy(np.log(prior[0])).type(torch.FloatTensor))
                #print(priorL.shape)
                priorL = priorL.cuda()
                priorR = Variable(torch.from_numpy(np.log(prior[1])).type(torch.FloatTensor))
                priorR = priorR.cuda()
            elif opts.out_mode == 'L' or opts.out_mode == 'R':
                prior = Variable(torch.from_numpy(np.log(prior)).type(torch.FloatTensor))
                prior = prior.cuda()
        for pr_batch,sample in enumerate(prod_dataloader):
            pr_x = Variable(sample['image'], volatile=True)
            pr_ids = sample['id']
            if opts.use_gpu:
                pr_x = pr_x.cuda()
            pr_y_gen = nnG(pr_x)
            if opts.save_prob_mat:
                for idx,data in enumerate(pr_y_gen.data):
                    fh = open(res_path + '/prob_mat/' + pr_ids[idx] + '.pickle', 'w')
                    pickle.dump(data.cpu().float().numpy(),fh,-1)
                    fh.close
            if opts.net_out_type == 'C':
                if opts.out_mode == 'LR':
                    if opts.do_prior:
                        pr_y_gen[0].data = pr_y_gen[0].data+priorL.data
                        pr_y_gen[1].data = pr_y_gen[1].data+priorR.data
                    _, pr_l = torch.max(pr_y_gen[0],dim=1,keepdim=True)
                    _, pr_r = torch.max(pr_y_gen[1],dim=1,keepdim=True)
                    pr_y_gen = torch.cat([pr_l, pr_r],1)
                elif opts.out_mode == 'L' or opts.out_mode == 'R':
                    if opts.do_prior:
                        pr_y_gen.data = pr_y_gen.data + prior.data
                    _, pr_y_gen = torch.max(pr_y_gen,dim=1,keepdim=True)
                else:
                    pass
            elif opts.net_out_type == 'R':
                pass
            else:
                pass
            for idx,data in enumerate(pr_y_gen.data):
                #--- TODO: update this function to proccess C-dim tensors at GPU
                pr_data.gen_page(pr_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)
        prod_end_time = time.time()
        logger.info('Production stage done. total time taken: {}'.format(prod_end_time-prod_start_time))
        logger.info('Average time per page: {}'.format((prod_end_time-prod_start_time)/prod_data.__len__()))

    logger.info('All Done...')
                

if __name__=='__main__':
    main()
