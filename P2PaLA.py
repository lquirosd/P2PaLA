from __future__ import print_function
from __future__ import division
#from builtins import range

import logging
import sys
import os
import shutil
import numpy as np
import torch
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.optparse import arguments
from nn_models import models
from data import dataset
from data import preprocessing as dp

loss_dic = {'L1':torch.nn.L1Loss(size_average=True),
            'MSE':torch.nn.MSELoss(size_average=False),
            'smoothL1':torch.nn.SmoothL1Loss(size_average=True)}


def save_checkpoint(state, is_best, opts, logger):
    """
    Save current model to checkpoints dir
    """
    #--- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    torch.save(state, opts.checkpoints + '/checkpoint.pth.tar')
    logger.info('Checkpoint saved to {}'.format(opts.checkpoints +
                '/checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(opts.checkpoints + '/checkpoint.pth.tar', 
                        opts.checkpoints + '/best_under' +
                        opts.best_criterion +
                        'criterion.pth.tar')
        logger.info('Best model saved to {}'.format(opts.checkpoints + 
                                                    '/best_under' + 
                                                    opts.best_criterion +
                                                    'criterion.pth.tar'))


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
    #--- init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    #--- keep this logger at DEBUG level, until aguments are processed 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #--- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    if check_inputs(opts,logger):
        logger.critical('Execution aborted due input errors...')
        exit(1)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(opts.log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #--- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    logger.debug(in_args)
    #--- configure TensorBoard display
    if opts.no_display:
        writer = ''
    else:
        import socket
        from datetime import datetime
        if opts.use_global_log:
            run_dir = opts.use_global_log
        else:
            run_dir = os.path.join(opts.work_dir, 'runs')
        log_dir = os.path.join(run_dir, 
                               datetime.now().strftime('%b%d_%H-%M-%S')+
                               '_'+socket.gethostname()+ opts.log_comment)
                                    
        writer = SummaryWriter(log_dir=log_dir) 
        logger.info('TensorBoard log will be stored at {}'.format(log_dir))
        logger.info('run: tensorboard --logdir {}'.format(run_dir))
    
    #--- Build transforms
    if opts.flip_img:
        transform = transforms.Compose([dataset.randomFlip(axis=2, prob=0.5),
                                        dataset.toTensor()])
    else:
        transform = transforms.Compose([dataset.toTensor()])
    opts.img_size = np.array(opts.img_size, dtype=np.int)
    if opts.do_train:
        #--- Get Train Data
        if opts.tr_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.tr_data))
            (opts.tr_img_list, opts.tr_label_list) = dp.htrDataProcess(
                                                        opts.tr_data,
                                                        opts.img_size,
                                                        opts.work_dir + '/data/train/',
                                                        opts.regions_colors,
                                                        line_width=opts.line_width,
                                                        line_color=opts.line_color,
                                                        processes=opts.num_workers,
                                                        logger=logger)

        train_data = dataset.htrDataset(img_lst=opts.tr_img_list,
                                        label_lst=opts.tr_label_list,
                                        transform=transform)
        train_dataloader = DataLoader(train_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)
        #--- Get Val data, if needed
        if opts.do_val:
            if opts.val_img_list == '':
                logger.info('Preprocessing data from{}'.format(opts.val_data))
                (opts.val_img_list, opts.val_label_list) = dp.htrDataProcess(
                                                        opts.val_data,
                                                        opts.img_size,
                                                        opts.work_dir + '/data/val/',
                                                        opts.regions_colors,
                                                        line_width=opts.line_width,
                                                        line_color=opts.line_color,
                                                        processes=opts.num_workers,
                                                        logger=logger)

            val_data = dataset.htrDataset(img_lst=opts.val_img_list,
                                          label_lst=opts.val_label_list,
                                          transform=transform)
            val_dataloader = DataLoader(val_data,
                                        batch_size=opts.batch_size,
                                        shuffle=opts.shuffle_data,
                                        num_workers=opts.num_workers,
                                        pin_memory=opts.pin_memory)

        #--- Build Models
        nnG = models.buildUnet(opts.input_channels,
                               opts.output_channels,
                               ngf=opts.cnn_ngf)
        #--- TODO: create a funtion @ models to define loss function
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
                logger.warning('Using {} loss funtion to resume training...'.format(opts.g_loss,))
            if opts.use_gpu:
                nnG = nnG.cuda()
                lossG = lossG.cuda()
        else:
            #--- send to GPU before init weigths
            if opts.use_gpu:
                nnG = nnG.cuda()
                lossG = lossG.cuda()
            nnG.apply(models.weights_init_normal)
        
        if opts.use_gan:
            nnD = models.buildGAN(opts.input_channels,
                                  opts.output_channels,
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
            else:
                if opts.use_gpu:
                    nnD = nnD.cuda()
                    lossD = lossD.cuda()
                nnD.apply(models.weights_init_normal) 

        #--- Do the actual train
        for epoch in xrange(opts.epochs):
            for b,sample in enumerate(train_dataloader):
                #--- Reset Grads
                optimizerG.zero_grad()
                x = Variable(sample['image'])
                y_gt = Variable(sample['label'])
                if opts.use_gpu:
                    x = x.cuda()
                    y_gt = y_gt.cuda()
                #--- scale out to [0-1]
                y_gen = (nnG(x)+1)*0.5
                g_loss = lossD(y_gen,y_gt)
                if opts.use_gan:
                    optimizerD.zero_grad()
                    real_D = torch.cat([x,y_gt],1)
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
                    d_loss_real.backward()
                    d_loss_fake.backward()
                    g_loss = (d_loss.data[0] + (opts.loss_lambda * g_loss))
                    optimizerD.step()
                g_loss.backward()
                optimizerG.step()
            #--- save current model, to test load func
            state = {
                    'nnG_state':            nnG.state_dict(),
                    'nnG_optimizer_state':  optimizerG.state_dict(),
                    'g_loss':               opts.g_loss
                    }
            if opts.use_gan:
                state['nnD_state'] =            nnD.state_dict()
                state['nnD_optimizer_state'] =  optimizerD.state_dict()
            save_checkpoint(state, False, opts, logger)
            #if best model under some criteria:
            #    best_model_wts = model.state_dict()
            # this will save the state dic to best_model wts
            # then we cam save it or restore the model to that state




if __name__=='__main__':
    main()
