from __future__ import print_function
from __future__ import division
from builtins import range

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# ------------------------------------------------------------------------------
# -------------      BEGIN UNET DEFINITION
# ------------------------------------------------------------------------------


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    Borrowed from: https://github.com/pytorch/pytorch/issues/3223

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(
        tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes)
    )


class buildUnet(nn.Module):
    """
    doc goes here :)
    """

    def __init__(
        self, input_nc, output_nc, ngf=64, net_type="R", out_mode=None
    ):
        super(buildUnet, self).__init__()
        # self.gpu_ids = gpu_ids

        model = uSkipBlock(
            ngf * 8,
            ngf * 8,
            ngf * 8,
            inner_slave=None,
            block_type="center",
            i_id="center",
        )
        model = uSkipBlock(
            ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id="a_1", useDO=True
        )
        model = uSkipBlock(
            ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id="a_2", useDO=True
        )
        model = uSkipBlock(
            ngf * 8, ngf * 8, ngf * 8, inner_slave=model, i_id="a_3"
        )
        # model = uSkipBlock(ngf*8, ngf*8, ngf*8, inner_slave=model, i_id='a_4')

        model = uSkipBlock(
            ngf * 4, ngf * 8, ngf * 4, inner_slave=model, i_id="a_5"
        )
        model = uSkipBlock(
            ngf * 2, ngf * 4, ngf * 2, inner_slave=model, i_id="a_6"
        )
        model = uSkipBlock(ngf, ngf * 2, ngf, inner_slave=model, i_id="a_7")
        # --- define output layer
        model = uSkipBlock(
            input_nc,
            ngf,
            output_nc,
            inner_slave=model,
            block_type=net_type,
            out_mode=out_mode,
            i_id="out",
        )
        # ---keep model
        self.model = model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    def forward(self, input_x):
        """
        ;)
        """
        # --- parallelize if GPU available and inputs are float
        # if self.gpu_ids and isinstance(input_x.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input_x, self.gpu_ids)
        # else:
        #    return self.model(input_x)
        return self.model(input_x)


class uSkipBlock(nn.Module):
    """
    """

    def __init__(
        self,
        input_nc,
        inner_nc,
        output_nc,
        inner_slave,
        block_type="inner",
        out_mode=None,
        i_id="0",
        useDO=False,
    ):
        super(uSkipBlock, self).__init__()
        self.type = block_type
        self.name = str(input_nc) + str(inner_nc) + str(output_nc) + self.type
        self.id = i_id
        self.out_mode = out_mode
        # self.output_nc = 2 * output_nc
        # --- TODO: move nn.Tanh to FW, then if/else R/C is not necessary
        if self.type == "R":
            # --- Handle out block
            e_conv = nn.Conv2d(
                input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_conv = nn.ConvTranspose2d(
                2 * inner_nc,
                output_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_non_lin = nn.ReLU(True)
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv, nn.Tanh()]
        elif self.type == "C":
            # --- handle out block, classification encoding
            e_conv = nn.Conv2d(
                input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_conv = nn.ConvTranspose2d(
                2 * inner_nc,
                output_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_non_lin = nn.ReLU(True)
            # model = [e_conv] + [inner_slave] + [d_non_lin, d_conv, nn.Softmax2d()]
            model = [e_conv] + [inner_slave] + [d_non_lin, d_conv]

        elif self.type == "center":
            # --- Handle center case
            e_conv = nn.Conv2d(
                input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            e_non_lin = nn.LeakyReLU(0.2, True)
            d_conv = nn.ConvTranspose2d(
                inner_nc,
                output_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [
                e_non_lin,
                e_conv,
                d_non_lin,
                d_conv,
                d_norm,
                nn.Dropout(0.5),
            ]
        elif self.type == "inner":
            # --- Handle internal case
            e_conv = nn.Conv2d(
                input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            e_non_lin = nn.LeakyReLU(0.2, True)
            e_norm = nn.BatchNorm2d(inner_nc)
            d_conv = nn.ConvTranspose2d(
                2 * inner_nc,
                output_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            d_non_lin = nn.ReLU(True)
            d_norm = nn.BatchNorm2d(output_nc)
            model = [
                e_non_lin,
                e_conv,
                e_norm,
                inner_slave,
                d_non_lin,
                d_conv,
                d_norm,
            ]
            if useDO:
                model = model + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, input_x):
        """
        """
        if self.type == "R":
            return self.model(input_x)
        elif self.type == "C":
            if self.out_mode == "L" or self.out_mode == "R":
                # if self.training:
                #    #return torch.log(self.model(input_x))
                #    return F.log_softmax(self.model(input_x),dim=1)
                # else:
                #    #return self.model(input_x)
                #    return F.softmax(self.model(input_x),dim=1)
                return F.log_softmax(self.model(input_x), dim=1)
            elif self.out_mode == "LR":
                # if self.training:
                #    #l_x = torch.log(self.model(input_x))
                #    #d_size = l_x.size(1)
                #    #return size_splits(l_x,[2,d_size-2], dim=1)
                #    x = self.model(input_x)
                #    l,r = size_splits(x,[2,x.size(1)-2],dim=1)
                #    return (F.log_softmax(l,dim=1),F.log_softmax(r,dim=1))
                # else:
                #    #l_x = self.model(input_x)
                #    #d_size = l_x.size(1)
                #    #return size_splits(l_x,[2,d_size-2], dim=1)
                #    x = self.model(input_x)
                #    l,r = size_splits(x,[2,x.size(1)-2],dim=1)
                #    return (F.softmax(l,dim=1),F.softmax(r,dim=1))
                x = self.model(input_x)
                l, r = size_splits(x, [2, x.size(1) - 2], dim=1)
                return (F.log_softmax(l, dim=1), F.log_softmax(r, dim=1))
            else:
                pass
        else:
            # --- send input fordward to next block
            return torch.cat([input_x, self.model(input_x)], 1)


# ------------------------------------------------------------------------------
# -------------      END UNET DEFINITION
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# -------------      BEGIN ADVERSARIAL D NETWORK
# ------------------------------------------------------------------------------


class buildDNet(nn.Module):
    """
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_layers=3):
        """
        """
        super(buildDNet, self).__init__()
        model = [
            nn.Conv2d(
                input_nc + output_nc,
                ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        ]
        model = model + [nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_prev = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model = model + [
                nn.Conv2d(
                    ngf * nf_prev,
                    ngf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model = model + [
            nn.Conv2d(
                ngf * nf_prev,
                ngf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                ngf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*model)
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    def forward(self, input_x):
        """
        """
        return self.model(input_x)


# ------------------------------------------------------------------------------
# -------------      END ADVERSARIAL DEFINITION
# ------------------------------------------------------------------------------


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.uniform_(m.weight.data, 0.0, 0.02)
        # init.constant(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        # init.constant(m.bias.data, 0.0)


def zero_bias(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.constant(m.bias.data, 0.0)


def off_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        m.p = 0.0
