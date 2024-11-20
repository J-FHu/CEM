import os
import time
import math
import numpy as np
import torch
from PIL import Image

from ModelZoo.utils import PIL2Tensor, _add_batch_one, mod_crop


class Logger(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.dataset_size = len(dataloader)
        self.first_epoch = 1
        self.total_epochs = config.niter + config.niter_decay
        self.epoch_iter = 0  # iter number within each epoch
        self.iter_record_path = os.path.join(self.config.checkpoints_dir, self.config.name, 'iter.txt')
        if config.isTrain and config.continue_train:
            try:
                self.first_epoch, self.epoch_iter = np.loadtxt(self.iter_record_path, delimiter=',', dtype=int)
                print('Resuming from epoch %d at iteration %d' % (self.first_epoch, self.epoch_iter))
            except:
                print('Could not load iteration record at %s. Starting from beginning.' %
                      self.iter_record_path)

        self.total_steps_so_far = (self.first_epoch - 1) * self.dataset_size + self.epoch_iter

        # return the iterator of epochs for the training

    def load_test_set(self, test_set_dir, scale):
        test_images = os.listdir(test_set_dir)
        self.testset = []
        self.testset_lr = []
        for img in test_images:
            pil_image = Image.open(os.path.join(test_set_dir, img))
            pil_w_o, pil_h_o = pil_image.size
            pil_image = pil_image.crop((0, 0, pil_w_o - pil_w_o % scale, pil_h_o - pil_h_o % scale))
            pil_w, pil_h = pil_image.size
            self.testset.append(_add_batch_one(
                PIL2Tensor(pil_image)
            ))
            self.testset_lr.append(_add_batch_one(
                PIL2Tensor(pil_image.resize((int(pil_w / scale), int(pil_h / scale)), Image.BICUBIC))
            ).cuda())

    def test_psnr(self, model):
        psnr = []
        with torch.no_grad():
            for i, img in enumerate(self.testset_lr):
                sr = model(img)
                psnr.append(self._psnr(sr.detach().cpu(), self.testset[i]))
        return sum(psnr) / len(psnr)


    def _psnr(self, sr, hr):
        # img1 and img2 have range [0, 255]
        img1 = sr.numpy().astype(np.float64) * 255
        img2 = hr.numpy().astype(np.float64) * 255
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def record_one_iteration(self):
        current_time = time.time()

        # the last remaining batch is dropped (see data/__init__.py),
        # so we can assume batch size is always opt.batchSize
        self.time_per_iter = (current_time - self.last_iter_time) / self.config.batch_size
        self.last_iter_time = current_time
        # self.total_steps_so_far += self.config.batch_size
        # self.epoch_iter += self.config.batch_size
        self.total_steps_so_far += 1
        self.epoch_iter += 1

    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print('End of epoch %d / %d, total iters: %d \t Time Taken: %d sec' %
              (self.current_epoch, self.total_epochs, self.total_steps_so_far, self.time_per_epoch))

    def needs_saving(self):
        return (self.total_steps_so_far % self.config.save_latest_freq) == 0

    def needs_printing(self):
        return (self.total_steps_so_far % self.config.print_freq) == 0

    def needs_testing(self):
        return ((self.current_epoch - 1) % self.config.test_freq) == 0

    def needs_displaying(self):
        return (self.total_steps_so_far % self.config.display_freq) == 0

