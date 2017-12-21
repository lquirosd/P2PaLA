from __future__ import print_function
from __future__ import division
#from builtins import range

import logging
import sys
import os
import time
import shutil
import numpy as np
import cv2
import errno

import torch
from torchvision import transforms
from torchvision import  utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.optparse import Arguments as arguments
from nn_models import models
from data import dataset
from data import imgprocess as dp

#--- reduce option isn't supported until pytorch 0.3.*
#--- TODO: install v0.3.0 or implement L1loss by myself
loss_dic = {'L1':torch.nn.L1Loss(size_average=False),#reduce=False),
            'MSE':torch.nn.MSELoss(size_average=True),
            'smoothL1':torch.nn.SmoothL1Loss(size_average=True)}

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

def save_checkpoint(state, is_best, opts, logger, epoch):
    """
    Save current model to checkpoints dir
    """
    #--- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    out_file = os.path.join(opts.checkpoints, 'checkpoint.pth.tar')
    torch.save(state, out_file)
    logger.info('Checkpoint saved to {} at epoch {}'.format(out_file, str(epoch)))
    if is_best:
        best_file = os.path.join(opts.checkpoints,
                    "".join(['best_under',opts.best_criterion,'criterion.pth.tar']))
        shutil.copyfile(out_file, best_file)
        logger.info('Best model saved to {} at epoch {}'.format(best_file, str(epoch)))

#--- TODO check all related to label_w
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
    #--- Init torch random 
    #--- This two are suposed to be merged in the future, for now keep boot
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    #--- Init model variable
    nnG = None
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
    
        #--- Build transforms
        if opts.flip_img:
            transform = transforms.Compose([dataset.randomFlip(axis=2, prob=0.5),
                                            dataset.toTensor()])
        else:
            transform = transforms.Compose([dataset.toTensor()])
        #--- Get Train Data
        if opts.tr_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.tr_data))
            tr_data = dp.htrDataProcess(
                                         opts.tr_data,
                                         opts.img_size,
                                         os.path.join(opts.work_dir,'data','train'),
                                         opts.regions_colors,
                                         line_width=opts.line_width,
                                         line_color=opts.line_color,
                                         processes=opts.num_workers,
                                         only_lines=opts.output_channels == 1,
                                         opts=opts,
                                         logger=logger)
            tr_data.pre_process()
            opts.tr_img_list = tr_data.img_list
            opts.tr_label_list = tr_data.label_list
            opts.tr_w_list = tr_data.w_list

        train_data = dataset.htrDataset(img_lst=opts.tr_img_list,
                                        label_lst=opts.tr_label_list,
                                        w_lst=opts.tr_w_list,
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
                va_data = dp.htrDataProcess(
                                             opts.val_data,
                                             opts.img_size,
                                             os.path.join(opts.work_dir,'data','val/'),
                                             opts.regions_colors,
                                             line_width=opts.line_width,
                                             line_color=opts.line_color,
                                             processes=opts.num_workers,
                                             only_lines=opts.output_channels == 1,
                                             opts=opts,
                                             logger=logger)
                va_data.pre_process()
                opts.val_img_list = va_data.img_list
                opts.val_label_list = va_data.label_list
                opts.val_w_list = va_data.w_list

            val_data = dataset.htrDataset(img_lst=opts.val_img_list,
                                          label_lst=opts.val_label_list,
                                          w_lst=opts.val_w_list,
                                          transform=transform)
            val_dataloader = DataLoader(val_data,
                                        batch_size=opts.batch_size,
                                        shuffle=False,
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
        #--- TODO: save model under "best" criterion 
        #--- TODO: compute statistical boostrap to define if a model is
        #---    statistically better than previous
        best_val = np.inf
        best_tr = np.inf
        for epoch in xrange(opts.epochs):
            epoch_start = time.time()
            epoch_lossG = 0
            epoch_lossGAN = 0
            epoch_lossR = 0
            epoch_lossD = 0
            for batch,sample in enumerate(train_dataloader):
                #--- Reset Grads
                optimizerG.zero_grad()
                x = Variable(sample['image'])
                y_gt = Variable(sample['label'])
                #w = Variable(sample['w'], requires_grad=False)
                if opts.use_gpu:
                    x = x.cuda()
                    y_gt = y_gt.cuda()
                    #w = w.cuda()
                y_gen = nnG(x)
                g_loss = lossG(y_gen,y_gt)
                #g_loss = torch.mul(g_loss, w)
                #g_loss = torch.sum(g_loss)
                g_loss = g_loss * (1/y_gen.data[0].numel())
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
                    epoch_lossD += d_loss.data[0]
                    #d_loss_real.backward()
                    #d_loss_fake.backward()
                    d_loss.backward()
                    optimizerD.step()
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
                    v_loss = lossG(v_y, v_label)
                    v_loss = v_loss * (1/v_y.data[0].numel())
                    val_loss += v_loss.data[0] / v_y.data.size()[0]
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
                save_checkpoint(state, False, opts, logger, epoch)
                if not opts.no_display:
                    dim = y_gen.data.size()
                    ex_dim = torch.ones(dim[0],3-dim[1],dim[2],dim[3])
                    if opts.use_gpu:
                        ex_dim = ex_dim.cuda()
                    o = torch.cat((y_gen.data,ex_dim),dim=1)
                    o = vutils.make_grid(o, normalize=False, scale_each=True)
                    t = torch.cat((y_gt.data,ex_dim),dim=1)
                    t = vutils.make_grid(t, normalize=False, scale_each=True)
                    writer.add_image('train/G_out', o, epoch)
                    writer.add_image('train/GT', t, epoch)
                    if opts.do_val:
                        v_dim = v_y.data.size()
                        v_ex_dim = torch.ones(v_dim[0],3-v_dim[1],v_dim[2],v_dim[3])
                        if opts.use_gpu:
                            v_ex_dim = v_ex_dim.cuda()
                        v_o = torch.cat((v_y.data,v_ex_dim),dim=1)
                        v_o = vutils.make_grid(v_o, normalize=False, scale_each=True)
                        writer.add_image('val/G_out', v_o, epoch)
                        v_t = torch.cat((v_label.data,v_ex_dim),dim=1)
                        v_t = vutils.make_grid(v_t, normalize=False, scale_each=True)
                        writer.add_image('val/GT', v_t, epoch)
        logger.info('Trining stage done. total time taken: {}'.format(time.time()-train_start))
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
            #--- Set model to eval, to perform inference step 
            nnG.eval()
            for v_batch,v_sample in enumerate(val_dataloader):
                #--- set vars to volatile, since bo backward used
                v_img = Variable(v_sample['image'], volatile=True)
                v_label = Variable(v_sample['label'], volatile=True)
                v_ids = v_sample['id']
                if opts.use_gpu:
                    v_img = v_img.cuda()
                    v_label = v_label.cuda()
                v_y = nnG(v_img)
                #--- save out as image for visual check
                for idx,data in enumerate(v_label.data):
                    img = tensor2img(data)
                    cv2.imwrite(os.path.join(res_path,
                                             'mask', v_ids[idx] +'_gt.png'),img)
                for idx,data in enumerate(v_y.data):
                    img = tensor2img(data)
                    cv2.imwrite(os.path.join(res_path,
                                             'mask', v_ids[idx] +'_out.png'),img)
                    va_data.gen_page(v_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)
        writer.add_graph(nnG, y_gen)
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
        logger.info('Results will be saved to {}'.format(res_path))

        if nnG == None:
            #--- Load Model 
            nnG = models.buildUnet(opts.input_channels,
                                   opts.output_channels,
                                   ngf=opts.cnn_ngf)
            logger.info('Resumming from model {}'.format(opts.prev_model))
            checkpoint = torch.load(opts.prev_model)
            nnG.load_state_dict(checkpoint['nnG_state'])
            if opts.use_gpu:
                nnG = nnG.cuda()
            nnG.eval()
            logger.debug('GEN Network:\n{}'.format(nnG)) 
            logger.debug('GEN Network, number of parameters: {}'.format(nnG.num_params))
        else:
            logger.debug('Using prevously loaded Generative module for test...')
            nnG.eval()

        #--- get test data
        if opts.te_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.te_data))
            te_data = dp.htrDataProcess(
                                         opts.te_data,
                                         opts.img_size,
                                         os.path.join(opts.work_dir,'data','test'),
                                         opts.regions_colors,
                                         line_width=opts.line_width,
                                         line_color=opts.line_color,
                                         processes=opts.num_workers,
                                         only_lines=opts.output_channels == 1,
                                         opts=opts,
                                         logger=logger)
            te_data.pre_process()
            opts.te_img_list = te_data.img_list
            opts.te_label_list = te_data.label_list
            opts.te_w_list = te_data.w_list
        
        transform = transforms.Compose([dataset.toTensor()])

        test_data = dataset.htrDataset(img_lst=opts.te_img_list,
                                        label_lst=opts.te_label_list,
                                        w_lst=opts.te_w_list,
                                        transform=transform)
        test_dataloader = DataLoader(test_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)
        for te_batch,sample in enumerate(test_dataloader):
            te_x = Variable(sample['image'], volatile=True)
            te_label = Variable(sample['label'], volatile=True)
            te_ids = sample['id']
            if opts.use_gpu:
                te_x = te_x.cuda()
                te_label = te_label.cuda()
            te_y_gen = nnG(te_x)
            for idx,data in enumerate(te_y_gen.data):
                #--- TODO: update this function to proccess C-dim tensors
                te_data.gen_page(te_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)
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
        logger.info('Results will be saved to {}'.format(res_path))

        if nnG == None:
            #--- Load Model 
            nnG = models.buildUnet(opts.input_channels,
                                   opts.output_channels,
                                   ngf=opts.cnn_ngf)
            logger.info('Resumming from model {}'.format(opts.prev_model))
            checkpoint = torch.load(opts.prev_model)
            nnG.load_state_dict(checkpoint['nnG_state'])
            if opts.use_gpu:
                nnG = nnG.cuda()
            nnG.eval()
            logger.debug('GEN Network:\n{}'.format(nnG)) 
            logger.debug('GEN Network, number of parameters: {}'.format(nnG.num_params))
        else:
            logger.debug('Using prevously loaded Generative module for prod...')
            nnG.eval()

        #--- get prod data
        if opts.prod_img_list == '':
            logger.info('Preprocessing data from {}'.format(opts.prod_data))
            pr_data = dp.htrDataProcess(
                                         opts.prod_data,
                                         opts.img_size,
                                         os.path.join(opts.work_dir,'data','prod'),
                                         opts.regions_colors,
                                         line_width=opts.line_width,
                                         line_color=opts.line_color,
                                         processes=opts.num_workers,
                                         only_lines=opts.output_channels == 1,
                                         build_labels=False,
                                         opts=opts,
                                         logger=logger)
            pr_data.pre_process()
            opts.prod_img_list = pr_data.img_list
        
        transform = transforms.Compose([dataset.toTensor()])

        prod_data = dataset.htrDataset(img_lst=opts.prod_img_list,
                                       transform=transform)
        prod_dataloader = DataLoader(prod_data,
                                      batch_size=opts.batch_size,
                                      shuffle=opts.shuffle_data,
                                      num_workers=opts.num_workers,
                                      pin_memory=opts.pin_memory)
        for pr_batch,sample in enumerate(prod_dataloader):
            pr_x = Variable(sample['image'], volatile=True)
            pr_ids = sample['id']
            if opts.use_gpu:
                pr_x = pr_x.cuda()
            pr_y_gen = nnG(pr_x)
            for idx,data in enumerate(pr_y_gen.data):
                #--- TODO: update this function to proccess C-dim tensors at GPU
                pr_data.gen_page(pr_ids[idx],
                                   data.cpu().float().numpy(),
                                   opts.regions,
                                   approx_alg=opts.approx_alg,
                                   num_segments=opts.num_segments,
                                   out_folder=res_path)

    logger.info('All Done...')
                

if __name__=='__main__':
    main()
