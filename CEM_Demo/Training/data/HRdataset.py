import numpy as np
import torch
import lmdb
import cv2
import os
import torch.utils.data as data
from collections import OrderedDict

from Training.utils.processing import read_img, modcrop, _read_img_lmdb
from Training.utils.data import get_image_paths


class InteractiveHR(data.Dataset):
    def __init__(self, config=None):
        super(InteractiveHR, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot,
                           config.edge_datatoot,
                           config.seg_datatoot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.3):
        """

        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def _crop_and_flip(self, tensor_list):
        H, W, C = tensor_list[0].shape
        assert H >= self.hr_size and W >= self.hr_size, 'Your input images must bigger than the image patch size'
        H_start = np.random.randint(0, H - self.hr_size)
        W_start = np.random.randint(0, W - self.hr_size)
        for tensor in tensor_list:
            tensor = tensor[H_start: H_start + self.hr_size, W_start: W_start + self.hr_size, :]
        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            for tensor in tensor_list:
                tensor = np.flipud(tensor)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            for tensor in tensor_list:
                tensor = np.fliplr(tensor)
        return tensor_list

    def _parse_Seg(self, Seg_img):
        pass

    def init_lmdb(self, HR_path, Edge_path, Seg_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_env = lmdb.open(HR_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.Edge_env = lmdb.open(Edge_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.Seg_env = lmdb.open(Seg_path,
                                  readonly=True,
                                  lock=False,
                                  readahead=False,
                                  meminit=False)
        self.paths_HR, self.sizes_HR = get_image_paths('lmdb', HR_path)
        self.paths_Edge, self.sizes_Edge = get_image_paths('lmdb', Edge_path)
        self.paths_Seg, self.sizes_Seg = get_image_paths('lmdb', Seg_path)
        assert len(self.paths_HR) == len(self.paths_Edge), 'Check the integrity of your dataset files'
        assert len(self.paths_HR) == len(self.paths_Seg), 'Check the integrity of your dataset files'

    def __getitem__(self, index):
        # get HR image
        HR_path = self.paths_HR[index]
        resolution_HR = [int(s) for s in self.paths_HR[index].split('_')]
        img_HR = read_img(self.HR_env, HR_path, resolution_HR)
        img_HR = modcrop(img_HR, self.sr_factor)  # Numpy float32, HWC, BGR, [0,1]"""

        # get Edge image
        Edge_path = self.paths_Edge[index]
        resolution_Edge = [int(s) for s in self.sizes_Edge[index].split('_')]
        img_Edge = read_img(self.Edge_env, Edge_path, resolution_Edge)  # Numpy float32, HWC, Gray, [0,1]"""

        # get
        Seg_path = self.paths_Seg[index]
        resolution_Seg = [int(s) for s in self.sizes_Seg[index].split('_')]
        img_Seg = _read_img_lmdb(self.Seg_env, Seg_path, resolution_Seg)  # Numpy int8, HWC, single channel, [0,255]"""

        # pre-processing



        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_GT = img_HR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()


        return None

    def __len__(self):
        return len(self.paths_HR)


class InteractiveEdgeHR(data.Dataset):
    def __init__(self, config=None):
        super(InteractiveEdgeHR, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot,
                           config.edge_datatoot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.5):
        """
        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def init_lmdb(self, HR_path, Edge_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_env = lmdb.open(HR_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.Edge_env = lmdb.open(Edge_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)

        self.paths_HR, self.sizes_HR = get_image_paths('lmdb', HR_path)
        self.paths_Edge, self.sizes_Edge = get_image_paths('lmdb', Edge_path)
        assert len(self.paths_HR) == len(self.paths_Edge), 'Check the integrity of your dataset files'
        self._check_resolution()

    def _check_resolution(self):
        print('Checking resolutions ', end='// ')
        count = 0
        for i in range(len(self.paths_HR)):
            resolution_HR = [int(s) for s in self.sizes_HR[i].split('_')]
            resolution_Edge = [int(s) for s in self.sizes_Edge[i].split('_')]
            if resolution_Edge[1:] != resolution_HR[1:]:
                count += 1
                print(f'{self.paths_HR[i]}, {resolution_Edge}, {resolution_HR}')
        print(f'Checked, find {count} images with wrong resolutions')

    def __getitem__(self, index):
        # get HR image
        HR_path = self.paths_HR[index]
        resolution_HR = [int(s) for s in self.sizes_HR[index].split('_')]
        img_HR = read_img(self.HR_env, HR_path, resolution_HR)
        # img_HR = modcrop(img_HR, self.sr_factor)
        # Numpy float32, HWC, BGR, [0,1]"""

        # get Edge image
        Edge_path = self.paths_Edge[index]
        resolution_Edge = [int(s) for s in self.sizes_Edge[index].split('_')]
        img_Edge = read_img(self.Edge_env, Edge_path, resolution_Edge)  # Numpy float32, HWC, Gray, [0,1]"""
        # img_Edge = modcrop(img_Edge, self.sr_factor)

        # pre-processing
        _, H, W = resolution_HR
        assert H >= self.hr_size and W >= self.hr_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.hr_size else np.random.randint(0, H - self.hr_size)
        W_start = 0 if W == self.hr_size else np.random.randint(0, W - self.hr_size)

        img_GT_c = img_HR[H_start: H_start + self.hr_size, W_start: W_start + self.hr_size, :]
        img_Edge_c = img_Edge[H_start: H_start + self.hr_size, W_start: W_start + self.hr_size, :]

        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            img_GT_c = np.flipud(img_GT_c)
            img_Edge_c = np.flipud(img_Edge_c)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            img_GT_c = np.fliplr(img_GT_c)
            img_Edge_c = np.fliplr(img_Edge_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT_c.shape[2] == 3:
            img_GT_c = img_GT_c[:, :, [2, 1, 0]]
        img_GT_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_c, (2, 0, 1)))).float()
        img_Edge_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Edge_c, (2, 0, 1)))).float()

        return img_GT_c, img_Edge_c

    def __len__(self):
        return len(self.paths_HR)


class HROnly(data.Dataset):
    def __init__(self, config=None):
        super(HROnly, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.5):
        """
        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def init_lmdb(self, HR_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_env = lmdb.open(HR_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.paths_HR, self.sizes_HR = get_image_paths('lmdb', HR_path)

    def __getitem__(self, index):
        # get HR image
        HR_path = self.paths_HR[index]
        resolution_HR = [int(s) for s in self.sizes_HR[index].split('_')]
        img_HR = read_img(self.HR_env, HR_path, resolution_HR)
        # Numpy float32, HWC, BGR, [0,1]"""

        # pre-processing
        _, H, W = resolution_HR
        assert H >= self.hr_size and W >= self.hr_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.hr_size else np.random.randint(0, H - self.hr_size)
        W_start = 0 if W == self.hr_size else np.random.randint(0, W - self.hr_size)

        img_GT_c = img_HR[H_start: H_start + self.hr_size, W_start: W_start + self.hr_size, :]

        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            img_GT_c = np.flipud(img_GT_c)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            img_GT_c = np.fliplr(img_GT_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT_c.shape[2] == 3:
            img_GT_c = img_GT_c[:, :, [2, 1, 0]]
        img_GT_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_c, (2, 0, 1)))).float()

        return img_GT_c

    def __len__(self):
        return len(self.paths_HR)


class HRLR(data.Dataset):
    def __init__(self, config=None):
        super(HRLR, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot,
                           config.lr_dataroot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.5):
        """
        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def init_lmdb(self, HR_path, LR_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_env = lmdb.open(HR_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.paths_HR, self.sizes_HR = get_image_paths('lmdb', HR_path)
        self.LR_env = lmdb.open(LR_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.paths_LR, self.sizes_LR = get_image_paths('lmdb', LR_path)

    def __getitem__(self, index):
        # get HR image
        HR_path = self.paths_HR[index]
        LR_path = self.paths_LR[index]
        assert HR_path == LR_path, 'check your lmdb keys'
        resolution_HR = [int(s) for s in self.sizes_HR[index].split('_')]
        resolution_LR = [int(s) for s in self.sizes_LR[index].split('_')]
        assert resolution_HR[1] == resolution_LR[1] * self.sr_factor, 'check your lmdb, the resolution is wrong'
        img_HR = read_img(self.HR_env, HR_path, resolution_HR)
        img_LR = read_img(self.LR_env, LR_path, resolution_LR)
        # Numpy float32, HWC, BGR, [0,1]"""

        # pre-processing
        _, H, W = resolution_LR
        assert H >= self.lr_size and W >= self.lr_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.lr_size else np.random.randint(0, H - self.lr_size)
        W_start = 0 if W == self.lr_size else np.random.randint(0, W - self.lr_size)

        img_LR_c = img_LR[H_start: H_start + self.lr_size, W_start: W_start + self.lr_size, :]
        img_HR_c = img_HR[H_start * self.sr_factor: H_start * self.sr_factor + self.hr_size, W_start * self.sr_factor: W_start * self.sr_factor + self.hr_size, :]

        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            img_LR_c = np.flipud(img_LR_c)
            img_HR_c = np.flipud(img_HR_c)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            img_LR_c = np.fliplr(img_LR_c)
            img_HR_c = np.fliplr(img_HR_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR_c.shape[2] == 3:
            img_LR_c = img_LR_c[:, :, [2, 1, 0]]
            img_HR_c = img_HR_c[:, :, [2, 1, 0]]
        img_LR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_c, (2, 0, 1)))).float()
        img_HR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_c, (2, 0, 1)))).float()

        return img_HR_c, img_LR_c

    def __len__(self):
        return len(self.paths_HR)


class HRLR_file(data.Dataset):
    def __init__(self, config=None):
        super(HRLR_file, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot,
                           config.lr_dataroot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.5):
        """
        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def init_lmdb(self, HR_path, LR_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_list = os.listdir(HR_path)
        self.HR_path = HR_path
        self.LR_list = os.listdir(LR_path)
        self.LR_path = LR_path
        assert len(self.HR_list) == len(self.LR_list)

    def __getitem__(self, index):
        # get HR image
        HR_name = self.HR_list[index]
        LR_name = self.HR_list[index]
        assert HR_name == LR_name, 'check your lmdb keys'
        img_HR = read_img(None, os.path.join(self.HR_path, HR_name))
        img_LR = read_img(None, os.path.join(self.LR_path, LR_name))
        # Numpy float32, HWC, BGR, [0,1]"""

        # pre-processing
        H, W, _ = img_LR.shape
        assert H >= self.lr_size and W >= self.lr_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.lr_size else np.random.randint(0, H - self.lr_size)
        W_start = 0 if W == self.lr_size else np.random.randint(0, W - self.lr_size)

        img_LR_c = img_LR[H_start: H_start + self.lr_size, W_start: W_start + self.lr_size, :]
        img_HR_c = img_HR[H_start * self.sr_factor: H_start * self.sr_factor + self.hr_size, W_start * self.sr_factor: W_start * self.sr_factor + self.hr_size, :]

        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            img_LR_c = np.flipud(img_LR_c)
            img_HR_c = np.flipud(img_HR_c)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            img_LR_c = np.fliplr(img_LR_c)
            img_HR_c = np.fliplr(img_HR_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR_c.shape[2] == 3:
            img_LR_c = img_LR_c[:, :, [2, 1, 0]]
            img_HR_c = img_HR_c[:, :, [2, 1, 0]]
        img_LR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_c, (2, 0, 1)))).float()
        img_HR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_c, (2, 0, 1)))).float()

        return img_HR_c, img_LR_c

    def __len__(self):
        return len(self.HR_list)



class HRLR_file_preload(data.Dataset):
    def __init__(self, config=None):
        super(HRLR_file_preload, self).__init__()
        if config is not None:
            self.initialize_dataset(config.lr_size,
                                    config.sr_factor,
                                    config.v_flip,
                                    config.h_flip)
            self.init_lmdb(config.hr_dataroot,
                           config.lr_dataroot)

    def initialize_dataset(self, lr_size, sr_factor, v_flip=0., h_flip=0.5):
        """
        :param lr_size:
        :param sr_factor:
        :param v_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :param h_flip: 0 for no flip, 0.3 for 30% image to be flipped, randomly
        :return:
        """
        self.lr_size = lr_size
        self.sr_factor = sr_factor
        self.hr_size = lr_size * sr_factor
        self.v_flip, self.h_flip = v_flip, h_flip

    def init_lmdb(self, HR_path, LR_path):
        # https://github.com/chainer/chainermn/issues/129
        self.HR_list = sorted(os.listdir(HR_path))
        self.HR_path = HR_path
        self.LR_list = sorted(os.listdir(LR_path))
        self.LR_path = LR_path

        self.HR_img_dict = OrderedDict()
        self.LR_img_dict = OrderedDict()
        for index, img in enumerate(self.HR_list):
            print('.', end='') if index % 10 == 0 else print(end='')
            HR_name = self.HR_list[index]
            LR_name = self.LR_list[index]
            assert HR_name[:4] == LR_name[:4], 'check your lmdb keys'
            img_HR = read_img(None, os.path.join(self.HR_path, HR_name))
            img_LR = read_img(None, os.path.join(self.LR_path, LR_name))
            self.HR_img_dict[index] = img_HR
            self.LR_img_dict[index] = img_LR
        print()


        assert len(self.HR_list) == len(self.LR_list)

    def __getitem__(self, index):
        img_HR = self.HR_img_dict[index]
        img_LR = self.LR_img_dict[index]
        # Numpy float32, HWC, BGR, [0,1]"""

        # pre-processing
        H, W, _ = img_LR.shape
        assert H >= self.lr_size and W >= self.lr_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.lr_size else np.random.randint(0, H - self.lr_size)
        W_start = 0 if W == self.lr_size else np.random.randint(0, W - self.lr_size)

        img_LR_c = img_LR[H_start: H_start + self.lr_size, W_start: W_start + self.lr_size, :]
        img_HR_c = img_HR[H_start * self.sr_factor: H_start * self.sr_factor + self.hr_size, W_start * self.sr_factor: W_start * self.sr_factor + self.hr_size, :]

        if self.v_flip and np.random.uniform(0, 1) < self.v_flip:
            img_LR_c = np.flipud(img_LR_c)
            img_HR_c = np.flipud(img_HR_c)
        if self.h_flip and np.random.uniform(0, 1) < self.h_flip:
            img_LR_c = np.fliplr(img_LR_c)
            img_HR_c = np.fliplr(img_HR_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR_c.shape[2] == 3:
            img_LR_c = img_LR_c[:, :, [2, 1, 0]]
            img_HR_c = img_HR_c[:, :, [2, 1, 0]]
        img_LR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_c, (2, 0, 1)))).float()
        img_HR_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_c, (2, 0, 1)))).float()

        return img_HR_c, img_LR_c

    def __len__(self):
        return len(self.HR_list)


