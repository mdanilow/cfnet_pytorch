""" Module containing the definitions of the networks used in training. Inspired
by https://github.com/adambielski/siamese-triplet/blob/master/networks.py , it
defines embedding networks, that represent the branch phase of the architectures,
and joining networks (siamese or triplet) that take as argument the embedding
networks.
"""
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_
from torchgeometry.image import get_gaussian_kernel2d

TEST_PATH = '/home/magister/CF_tracking/cfnet_pytorch/tesciki'


device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")


class Conv2net(nn.Module):

    def __init__(self, xavier_init=False):
        super(Conv2net, self).__init__()
        self.layers = nn.Squential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLu(),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 32, kernel_size=5, stride=1, groups=2, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLu()
        )


    def forward(self, x):
        return self.layers(x)


class BaselineEmbeddingNet(nn.Module):
    """ Definition of the embedding network used in the baseline experiment of
    Bertinetto et al in https://arxiv.org/pdf/1704.06036.pdf.
    It basically corresponds to the convolutional stage of AlexNet, with some
    of its hyperparameters changed.
    """

    def __init__(self, xavier_init=False):
        super(BaselineEmbeddingNet, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11,
                                                  stride=2, bias=True),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=2),

                                        nn.Conv2d(96, 256, kernel_size=5,
                                                  stride=1, groups=2,
                                                  bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=1),
                                        nn.Conv2d(256, 384, kernel_size=3,
                                                  stride=1, groups=1,
                                                  bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 384, kernel_size=3,
                                                  stride=1, groups=2,
                                                  bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 32, kernel_size=3,
                                                  stride=1, groups=2,
                                                  bias=True))
        
        if xavier_init:
            for module in self.fully_conv:
                weights_init(module)

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class VGG11EmbeddingNet_5c(nn.Module):
    """ Embedding branch based on Pytorch's VGG11 with Batchnorm (https://pytor
    ch.org/docs/stable/torchvision/models.html). This is version 5c, meaning
    that it has 5 convolutional layers, it follows the original model up until
    the 13th layer (The ReLU after the 4th convolution), in order to keep the
    total stride equal to 4. It adds the 5th convolutional layer which acts as
    a bottleck a feature bottleneck reducing the features from 256 to 32 and
    must always be trained. The layers 0 to 13 can be loaded from
    torchvision.models.vgg11_bn(pretrained=True)
    """

    def __init__(self):
        super(VGG11EmbeddingNet_5c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),

                                        nn.Conv2d(64, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3,
                                                  stride=1, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class VGG16EmbeddingNet_8c(nn.Module):
    """ Embedding branch based on Pytorch's VGG16 with Batchnorm (https://pytor
    ch.org/docs/stable/torchvision/models.html). This is version 8c, meaning
    that it has 8 convolutional layers, it follows the original model up until
    the 22th layer (The ReLU after the 7th convolution), in order to keep the
    total stride equal to 4. It adds the 8th convolutional layer which acts as
    a bottleck a feature bottleneck reducing the features from 256 to 32 and
    must always be trained. The layers 0 to 22 can be loaded from
    torchvision.models.vgg16_bn(pretrained=True)
    """

    def __init__(self):
        super(VGG16EmbeddingNet_8c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(64, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3,
                                                  stride=1, bias=True))


    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class CorrFilter(torch.autograd.Function):

    @staticmethod
    def forward(self, x, cf_target, lambd):

        n = cf_target.shape[0] * cf_target.shape[1]
        x_f = fft2(x)
        y_f = fft2(cf_target)
        k_f = lambd + torch.sum( torch.mul(torch.conj(x_f), x_f), 1 ) / n
        a_f = torch.mul(y_f, 1/k_f) / n
        w_f = torch.mul(torch.unsqueeze(torch.conj(a_f), 1), x_f)
        w = torch.real(ifft2(w_f))
        self.save_for_backward(x_f, a_f, k_f)

        return w

    @staticmethod
    def backward(self, grad_output):

        x_f, a_f, k_f = self.saved_tensors
        n = x_f.shape[2] * x_f.shape[3]
        der_w_f = fft2(grad_output)
        der_a_f = torch.sum( torch.mul(x_f, torch.conj(der_w_f)), 1)
        der_x_f = torch.mul(torch.unsqueeze(a_f, 1), der_w_f)
        
        der_k_f = torch.mul(-der_a_f, torch.conj(a_f / k_f))
        der_x_f = der_x_f + 2/n * torch.mul(torch.real(torch.unsqueeze(der_k_f, 1)), x_f)
        der_x = torch.real(ifft2(der_x_f))

        return der_x, None, None


class CFblock(nn.Module):

    def __init__(self, in_sz):
        super(CFblock, self).__init__()
        self.in_sz = in_sz
        self.sigma = 2
        self.lambd = 10 # regularization for cf
        self.crop_margin = 16
        self.window = self.generate_window()
        # self.cf_target = get_gaussian_kernel2d((in_sz, in_sz), (self.sigma, self.sigma)).to(device)
        self.cf_target = self.generate_gaussian(self.in_sz, self.sigma).to(device)
        # np.savetxt(join(TEST_PATH, 'targettorch.csv'), self.cf_target.cpu().detach(), delimiter=',')
        # np.savetxt(join(TEST_PATH, 'windowtorch.csv'), self.window.cpu().detach(), delimiter=',')
        # print('DONE')
        self.corr_filter = CorrFilter.apply


    def corr_filter_xd(self, x, cf_target, lambd):

        n = cf_target.shape[0] * cf_target.shape[1]
        x_f = fft2(x)
        y_f = fft2(cf_target)
        k_f = lambd + torch.sum( torch.mul(torch.conj(x_f), x_f), 1 ) / n
        a_f = torch.mul(y_f, 1/k_f) / n
        w_f = torch.mul(torch.unsqueeze(torch.conj(a_f), 1), x_f)
        w = torch.real(ifft2(w_f))

        return w


    def generate_window(self):
        onedim_win = torch.hann_window(self.in_sz, periodic=False)
        twodim_win = torch.mul(onedim_win.view(-1, 1), onedim_win.view(1, -1)).to(device)

        return twodim_win


    def generate_gaussian(self, size, sigma):

        xg, yg = np.meshgrid(range(size), range(size))
        half = size // 2
        xg = np.mod(xg + half, size) - half
        yg = np.mod(yg + half, size) - half

        y = np.exp(-(np.square(xg) + np.square(yg)) / (2 * sigma**2))
        
        return torch.tensor(y, dtype=torch.float32)

    
    def forward(self, x):

        x = torch.mul(x, self.window)
        x = self.corr_filter(x, self.cf_target, self.lambd)
        x = x[:, :, self.crop_margin:-self.crop_margin, self.crop_margin:-self.crop_margin]

        return x


class CFnet(nn.Module):

    def __init__(self, params, corr_map_size=33, stride=4):
        super(CFnet, self).__init__()
        self.embedding_net_reference = BaselineEmbeddingNet(xavier_init=True)
        self.embedding_net_search = BaselineEmbeddingNet(xavier_init=True)
        self.match_batchnorm = nn.BatchNorm2d(1)
        # TODO compute this shape!
        self.ref_feature_sz = 49
        self.cf_block = CFblock(self.ref_feature_sz)

        self.upscale = params.upscale
        # TODO calculate automatically the final size and stride from the
        # parameters of the branch
        self.corr_map_size = corr_map_size
        self.stride = stride
        # Calculates the upscale size based on the correlation map size and
        # the total stride of the network, so as to align the corners of the
        # original and the upscaled one, which also aligns the centers.
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        # The upscale_factor is the correspondence between a movement in the output
        # feature map and the input images. So if a network has a total stride of 4
        # and no deconvolutional or upscaling layers, a single pixel displacement
        # in the output corresponds to a 4 pixels displacement in the input
        # image. The easiest way to compensate this effect is to do a bilinear
        # or bicubic upscaling.
        if self.upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

        self.CFscale = nn.Parameter(torch.tensor(1.0))
        self.CFbias = nn.Parameter(torch.tensor(-0.5))


    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        # embedding_reference = self.embedding_net_reference(x1)
        # cf_template = self.cf_block(embedding_reference)
        # cf_template = torch.mul(self.CFscale, cf_template) + self.CFbias
        cf_template = self.get_template_embedding(x1)

        embedding_search = self.get_search_embedding(x2)
        match_map = self.match_corr(cf_template, embedding_search)
        return match_map


    def get_template_embedding(self, x):
        embedding_reference = self.embedding_net_reference(x)
        cf_template = self.cf_block(embedding_reference)
        cf_template = torch.mul(self.CFscale, cf_template) + self.CFbias

        return cf_template


    def get_search_embedding(self, x):

        return self.embedding_net_search(x)


    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b)
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
                                      align_corners=False)

        return match_map


class SiameseNetCrop(nn.Module):
    """ The basic siamese network joining network, that takes the outputs of
    two embedding branches and joins them applying a correlation operation.
    Should always be used with tensors of the form [B x C x H x W], i.e.
    you must always include the batch dimension.
    """

    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4):
        super(SiameseNetCrop, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.crop_margin = 16

        self.upscale = upscale
        # TODO calculate automatically the final size and stride from the
        # parameters of the branch
        self.corr_map_size = corr_map_size
        self.stride = stride
        # Calculates the upscale size based on the correlation map size and
        # the total stride of the network, so as to align the corners of the
        # original and the upscaled one, which also aligns the centers.
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        # The upscale_factor is the correspondence between a movement in the output
        # feature map and the input images. So if a network has a total stride of 4
        # and no deconvolutional or upscaling layers, a single pixel displacement
        # in the output corresponds to a 4 pixels displacement in the input
        # image. The easiest way to compensate this effect is to do a bilinear
        # or bicubic upscaling.
        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride


    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.get_template_embedding(x1)
        embedding_search = self.get_search_embedding(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    # def get_embedding(self, x):
    #     return self.embedding_net(x)

    def get_template_embedding(self, x):
        embedding_reference = self.embedding_net(x)
        # print('precrop ref shape:', embedding_reference.shape)
        embedding_reference = embedding_reference[:, :, self.crop_margin:-self.crop_margin, self.crop_margin:-self.crop_margin]
        # print('ref shape:', embedding_reference.shape)

        return embedding_reference


    def get_search_embedding(self, x):
        embedding_search = self.embedding_net(x)
        return embedding_search


    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b)
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
                                      align_corners=False)

        return match_map


class SiameseNet(nn.Module):
    """ The basic siamese network joining network, that takes the outputs of
    two embedding branches and joins them applying a correlation operation.
    Should always be used with tensors of the form [B x C x H x W], i.e.
    you must always include the batch dimension.
    """

    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)

        self.upscale = upscale
        # TODO calculate automatically the final size and stride from the
        # parameters of the branch
        self.corr_map_size = corr_map_size
        self.stride = stride
        # Calculates the upscale size based on the correlation map size and
        # the total stride of the network, so as to align the corners of the
        # original and the upscaled one, which also aligns the centers.
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        # The upscale_factor is the correspondence between a movement in the output
        # feature map and the input images. So if a network has a total stride of 4
        # and no deconvolutional or upscaling layers, a single pixel displacement
        # in the output corresponds to a 4 pixels displacement in the input
        # image. The easiest way to compensate this effect is to do a bilinear
        # or bicubic upscaling.
        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)

    def get_template_embedding(self, x):
        return self.embedding_net(x)

    def get_search_embedding(self, x):
        return self.embedding_net(x)

    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b)
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
                                      align_corners=False)

        return match_map


def weights_init(model):
    """ Initializes the weights of the CNN model using the Xavier
    initialization.
    """
    if isinstance(model, nn.Conv2d):
        xavier_uniform_(model.weight, gain=math.sqrt(2.0))
        constant_(model.bias, 0.1)
    elif isinstance(model, nn.BatchNorm2d):
        normal_(model.weight, 1.0, 0.02)
        zeros_(model.bias)
