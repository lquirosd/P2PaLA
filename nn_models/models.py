from __future__ import print_function
from __future__ import division

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init

#------------------------------------------------------------------------------
#-------------      BEGIN UNET DEFINITION
#------------------------------------------------------------------------------

class buildUnet(nn.Module):
    """
    doc goes here :)
    """
    def __init__(self,input_nc,output_nc,ngf=64, use_class=False):
        super(buildUnet,self).__init__()
        #self.gpu_ids = gpu_ids
        
        model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=None, block_type='center',i_id='center')
        model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=model, i_id='a_1',useDO=True)
        model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=model, i_id='a_2',useDO=True)
        model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=model, i_id='a_3')
        #model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=model, i_id='a_4')
        
        model = uSkipBlock(ngf*4, ngf*8, ngf*4, inner_slave=model, i_id='a_5')
        model = uSkipBlock(ngf*2, ngf*4, ngf*2, inner_slave=model, i_id='a_6')
        model = uSkipBlock(ngf  , ngf*2, ngf  , inner_slave=model, i_id='a_7')
        if use_class:
            #--- TODO: Update to separate lines and regions
            #--- this is a test, so only lines are supported
            model = uSkipBlock(input_nc, ngf, output_nc+1, inner_slave=model, block_type='class_out', i_id='out')
        else:
            model = uSkipBlock(input_nc, ngf, output_nc, inner_slave=model, block_type='reg_out', i_id='out')
        #---keep model
        self.model = model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()


    def forward(self, input_x):
        """
        ;)
        """
        #--- parallelize if GPU available and inputs are float
        #if self.gpu_ids and isinstance(input_x.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input_x, self.gpu_ids)
        #else:
        #    return self.model(input_x)
        return self.model(input_x)

class uSkipBlock(nn.Module):
    """
    """
    def __init__(self,input_nc,inner_nc,output_nc,inner_slave,block_type='inner',i_id='0',useDO=False):
        super(uSkipBlock,self).__init__()
        self.type = block_type
        self.name = str(input_nc) + str(inner_nc) + str(output_nc) + self.type
        self.id = i_id
        self.output_nc = 2 * output_nc
        if (self.type == 'reg_out'):
            #--- Handle out block
            e_conv = nn.Conv2d(input_nc,inner_nc,kernel_size=4,
                               stride=2,padding=1,bias=False)
            d_conv = nn.ConvTranspose2d(2*inner_nc, output_nc,kernel_size=4,
                                        stride=2,padding=1,bias=False)
            d_non_lin = nn.ReLU(True)
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv, nn.Tanh()]
        elif (self.type == 'class_out'):
            #--- handle out block, classification encoding
            e_conv = nn.Conv2d(input_nc,inner_nc,kernel_size=4,
                               stride=2,padding=1,bias=False)
            d_conv = nn.ConvTranspose2d(2*inner_nc, output_nc,kernel_size=4,
                                        stride=2,padding=1,bias=False)
            d_non_lin = nn.ReLU(True)
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv, nn.Softmax2d()]
            
        elif (self.type == 'center'):
            #--- Handle center case
            e_conv = nn.Conv2d(input_nc,inner_nc,kernel_size=4,
                               stride=2,padding=1,bias=False)
            e_non_lin = nn.LeakyReLU(0.2,True)
            d_conv = nn.ConvTranspose2d(inner_nc, output_nc,kernel_size=4,
                                        stride=2,padding=1,bias=False)
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [e_non_lin, e_conv, d_non_lin,d_conv,d_norm, nn.Dropout(0.5)]
        elif (self.type == 'inner'):
            #--- Handle internal case
            e_conv = nn.Conv2d(input_nc,inner_nc,kernel_size=4,
                               stride=2,padding=1,bias=False)
            e_non_lin = nn.LeakyReLU(0.2,True)
            e_norm = nn.BatchNorm2d(inner_nc)
            d_conv = nn.ConvTranspose2d(2 * inner_nc, output_nc,kernel_size=4,
                                        stride=2,padding=1,bias=False)
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [e_non_lin, e_conv, e_norm,
                     inner_slave,
                     d_non_lin,d_conv,d_norm]
            if useDO:
                model = model + [nn.Dropout(0.5)]
        
        self.model = nn.Sequential(*model)
    def forward(self,input_x):
        """
        """
        if (self.type == 'reg_out'):
            #--- TODO: handle paralellism over several GPUs
            return self.model(input_x)
        elif (self.type == 'class_out'):
            #--- TODO: check for numerical instability by log 
            if self.training:
                return torch.log(self.model(input_x))
            else:
                #--- remove log during inference 
                return self.model(input_x)
        else:
            #--- send input fordward to next block
            return torch.cat([input_x,self.model(input_x)], 1)

#------------------------------------------------------------------------------
#-------------      END UNET DEFINITION
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#-------------      BEGIN ADVERSARIAL D NETWORK
#------------------------------------------------------------------------------

class buildDNet(nn.Module):
    """
    """
    def __init__(self,input_nc,output_nc,ngf=64,n_layers=3):
        """
        """
        super(buildDNet,self).__init__()
        model = [nn.Conv2d(input_nc+output_nc,ngf,kernel_size=4,
                          stride=2,padding=1,bias=False)]
        model = model + [nn.LeakyReLU(0.2,True)]
        nf_mult = 1
        nf_prev = 1
        for n in xrange(1,n_layers):
            nf_prev = nf_mult
            nf_mult = min(2**n,8)
            model = model + [
                           nn.Conv2d(ngf*nf_prev,ngf*nf_mult,kernel_size=4,
                                     stride=2,padding=1,bias=False),
                           nn.BatchNorm2d(ngf*nf_mult),
                           nn.LeakyReLU(0.2,True)
                           ]
            
        nf_prev = nf_mult
        nf_mult = min(2**n_layers,8)
        model = model + [
                        nn.Conv2d(ngf*nf_prev,ngf*nf_mult,kernel_size=4,
                                 stride=1,padding=1,bias=False),
                        nn.BatchNorm2d(ngf*nf_mult),
                        nn.LeakyReLU(0.2,True),
                        nn.Conv2d(ngf*nf_mult,1,kernel_size=4,stride=1,
                                  padding=1,bias=False),
                        nn.Sigmoid()
                        ]
        self.model = nn.Sequential(*model)
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    def forward(self,input_x):
        """
        """
        return self.model(input_x)

#------------------------------------------------------------------------------
#-------------      END ADVERSARIAL DEFINITION
#------------------------------------------------------------------------------

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        #init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        #init.constant(m.bias.data, 0.0)

def zero_bias(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.constant(m.bias.data, 0.0)

def off_dropout(m):
    classname = m.__class__.__name__ 
    if classname.find('Dropout') != -1:
        m.p = 0.0
