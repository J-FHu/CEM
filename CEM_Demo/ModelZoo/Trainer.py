import os
from itertools import chain
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ModelZoo.utils import Tensor2PIL, PIL2Tensor

from ModelZoo import get_model


class BasicModel(object):
    """
    Basic model class for InterSRGAN
    """
    def __init__(self):
        self.Tensor = torch.FloatTensor
        self.cuda = False

    def _check_tensor(self, tensor):
        if isinstance(tensor, self.Tensor):
            return tensor
        else:
            if self.Tensor is torch.cuda.FloatTensor:
                return tensor.cuda()
            else:
                return tensor.cpu()

    def _parse_cuda(self, gpu_num):
        if torch.cuda.is_available() and torch.cuda.device_count() >= len(gpu_num):
            pass
        else:
            raise Exception('Check your gpu number and cuda env.')


class SR(BasicModel):
    def __init__(self):
        super(SR, self).__init__()
        self.networks = []
        self.losses = []
        self.phase = 'test'

    def init_model(self, config):
        self._build_model(config.model_name,
                          SR_factor=config.sr_factor,
                          LR_channels=config.lr_channels)

    def _build_model(self, model_name, SR_factor, LR_channels):
        self.SR_factor = SR_factor
        self.LR_channels = LR_channels
        self.SRNet = get_model(model_name, factor=SR_factor, num_channels=LR_channels)
        self.D = None

    def _build_gan_model(self):
        pass

    def _build_sr_loss(self, sr_content='l1'):
        if sr_content == 'l1':
            self.SR_Loss = nn.L1Loss()
        elif sr_content == 'l2':
            self.SR_Loss = nn.MSELoss()
        else:
            raise NotImplementedError()
        self.losses.append(self.SR_Loss)

    def _init_sr_optim(self, SR_LR, optimizer='adam'):
        self.optim_D = None
        if optimizer == 'adam':
            self.optim_SR = optim.Adam(self.SRNet.parameters(), lr=SR_LR, betas=(self.config.beta1, self.config.beta2))
        elif optimizer == 'rmsprop':
            self.optim_SR = optim.RMSprop(self.SRNet.parameters(), lr=SR_LR)
        elif optimizer == 'sgd':
            self.optim_SR = optim.SGD(self.SRNet.parameters(), lr=SR_LR)
        else:
            raise NotImplementedError()

    def _move_to_gpu(self, gpu_num):
        self._parse_cuda(gpu_num)
        self.Tensor = torch.cuda.FloatTensor
        print(f'Using {len(gpu_num)} GPUs')
        self.SRNet = nn.DataParallel(self.SRNet)
        self.SRNet.cuda()
        print(f'Moving {type(self.SRNet.module).__name__} to GPU...')
        if self.phase == 'train':
            for loss in self.losses:
                loss.cuda()
        self.cuda = True

    def init_train(self, config):
        self.config = config
        if not self.SRNet:
            self.init_model(config)
        self.phase = 'train'
        self.old_lr = config.generator_lr
        self._build_sr_loss(config.sr_loss_mode)
        self._init_sr_optim(config.generator_lr,
                            config.optimizer)
        self._move_to_gpu(config.gpu_ids)
        self.gan = False

    def init_gan_train(self, config):
        pass

    def train_step(self, HR_images, LR_images):
        """
        :param HR_images: Tensor B * C * HR_H, HR_W
        :return:
        """
        HR_images = HR_images
        LR_bicubic = LR_images
        SR_bicubic = self.SRNet(LR_bicubic)

        sr_content_loss = self.SR_Loss(SR_bicubic, HR_images.cuda())

        self.optim_SR.zero_grad()
        sr_content_loss.backward()
        self.optim_SR.step()

        losses = OrderedDict([('sr_content_loss', sr_content_loss.item())])
        images_return = OrderedDict([('sr_bicubic', SR_bicubic.cpu()),
                                     ('lr_bicubic', LR_bicubic.cpu()),
                                     ('hr_images', HR_images.cpu())])

        return losses, images_return

    def update_learning_rate(self, epoch):
        if epoch > self.config.niter:
            decay_every = int(self.config.niter_decay / self.config.decay_round)
            decay_pow = (epoch - self.config.niter) // decay_every
            new_lr = self.config.generator_lr / (2 ** (decay_pow + 1))
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.config.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            if self.gan:
                for param_group in self.optim_D.param_groups:
                    param_group['lr'] = new_lr_D
            for param_group in self.optim_SR.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def train_gan_step(self):
        pass

    def sr(self, LR_images):
        """
        :param LR_images: LR images of size [B, C, LR_H, LR_W]
        :param condition: condition inputs
        :param encoder: Use Encoder or Not
        :param scales_return: True for return all scales
        :return:
        """
        LR_images = self._check_tensor(LR_images)
        SR_images = self.SRNet(LR_images)
        return SR_images

    def _save_name(self, epoch):
        return os.path.join(self.config.checkpoints_dir, f'{self.config.model_name}_D_{epoch}.pth'), \
               os.path.join(self.config.checkpoints_dir, f'{self.config.model_name}_{epoch}.pth')

    def load(self, epoch):
        D_name, G_name, = self._save_name(epoch)
        self.SRNet.load_state_dict(torch.load(G_name))
        if self.gan:
            self.D.load_state_dict(torch.load(D_name))

    def save(self, epoch):
        # save meta information
        D_name, G_name = self._save_name(epoch)
        torch.save(
            self.SRNet.module.state_dict() if self.cuda else self.SRNet.state_dict(),
            G_name)
        if self.gan:
            torch.save(
                self.D.module.state_dict() if self.cuda else self.D.state_dict(),
                D_name)

