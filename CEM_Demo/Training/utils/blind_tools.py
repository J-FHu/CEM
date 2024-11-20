import numpy as np
import random
import functools
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import random
from PIL import Image
from io import BytesIO


def randomJPEGcompression(image):
    qf = random.randrange(35, 60)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


jpeg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(randomJPEGcompression),
            transforms.ToTensor()
])


def batchRandomJPEGCompression(batch_images):
    image_list = torch.split(batch_images, batch_images.shape[0], dim=0)
    for i in range(len(image_list)):
        image_list[i] = jpeg_transform(image_list[i].squeeze(0))
        image_list[i] = image_list[i].unsqueeze(0)
    return torch.cat(image_list, dim=0)


def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.25, sig_max=4.0, scaling=3, l=15, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=1.5, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=15, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def random_batch_kernel(batch, l=21, sig_min=0.25, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


class BatchSRKernel(object):
    def __init__(self, l=15, sig_min=0.25, sig_max=4.0, rate_iso=0.3, scaling=3):
        self.l = l
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, batch, tensor=False):
        return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate, scaling=self.scaling, tensor=tensor)


def random_noise_level(high, rate_cln=0.3):
    if np.random.uniform() < rate_cln:
        return 0.0
    else:
        return np.random.uniform() * high


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


def random_batch_noise(batch, high, rate_cln=0.2):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


class Blur(nn.Module):
    def __init__(self, l=21, kernel=None):
        super(Blur, self).__init__()
        self.l = l
        self.pad = nn.ReflectionPad2d(l // 2)
        self.kernel = torch.FloatTensor(kernel).view((1, 1, self.l, self.l))

    def cuda(self, device=None):
        self.kernel = self.kernel.cuda()

    def forward(self, input):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        input_CBHW = pad.view((C * B, 1, H_p, W_p))

        return F.conv2d(input_CBHW, self.kernel).view(B, C, H, W)


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))

            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))

