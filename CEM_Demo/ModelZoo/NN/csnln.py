from ..NN import common
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def normalize(x):
    return x.mul_(2).add_(-1)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def make_model(args, parent=False):
    return CSNLN(args)


# in-scale non-local attention
class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(NonLocalAttention, self).__init__()
        self.conv_match1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0, 2, 1).view(N, -1, H, W)


# cross-scale non-local attention
class CrossScaleAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        # self.register_buffer('fuse_weight', fuse_weight)

    def forward(self, input):
        # get embedding
        embed_w = self.conv_assembly(input)
        match_input = self.conv_match_1(input)

        # b*c*h*w
        shape_input = list(embed_w.size())  # b*c*h*w
        input_groups = torch.split(match_input, 1, dim=0)
        # kernel size on input for matching
        kernel = self.scale * self.ksize

        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                      strides=[self.stride * self.scale, self.stride * self.scale],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling X to form Y for cross-scale matching
        ref = F.interpolate(input, scale_factor=1. / self.scale, mode='bilinear')
        ref = self.conv_match_2(ref)
        w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        shape_ref = ref.shape
        # w shape: [N, C, k, k, L]
        w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        y = []
        scale = self.softmax_scale
        # 1*1*k*k
        # fuse_weight = self.fuse_weight

        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            # normalize
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi

            # Compute correlation map
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]

            yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # (B=1, C=32*32, H=32, W=32)
            # rescale matching score
            yi = F.softmax(yi * scale, dim=1)
            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for reconsturction
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride * self.scale, padding=self.scale)

            yi = yi / 6.
            y.append(yi)

        y = torch.cat(y, dim=0)
        return y


# projection between attention branches
class MultisourceProjection(nn.Module):
    def __init__(self, in_channel, kernel_size=3, scale=2, conv=common.default_conv):
        super(MultisourceProjection, self).__init__()
        deconv_ksize, stride, padding, up_factor = {
            2: (6, 2, 2, 2),
            3: (9, 3, 3, 3),
            4: (6, 2, 2, 2)
        }[scale]
        self.up_attention = CrossScaleAttention(scale=up_factor)
        self.down_attention = NonLocalAttention()
        self.upsample = nn.Sequential(
            *[nn.ConvTranspose2d(in_channel, in_channel, deconv_ksize, stride=stride, padding=padding), nn.PReLU()])
        self.encoder = common.ResBlock(conv, in_channel, kernel_size, act=nn.PReLU(), res_scale=1)

    def forward(self, x):
        down_map = self.upsample(self.down_attention(x))
        up_map = self.up_attention(x)

        err = self.encoder(up_map - down_map)
        final_map = down_map + err

        return final_map


# projection with local branch
class RecurrentProjection(nn.Module):
    def __init__(self, in_channel, kernel_size=3, scale=2, conv=common.default_conv):
        super(RecurrentProjection, self).__init__()
        self.scale = scale
        stride_conv_ksize, stride, padding = {
            2: (6, 2, 2),
            3: (9, 3, 3),
            4: (6, 2, 2)
        }[scale]

        self.multi_source_projection = MultisourceProjection(in_channel, kernel_size=kernel_size, scale=scale,
                                                             conv=conv)
        self.down_sample_1 = nn.Sequential(
            *[nn.Conv2d(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding), nn.PReLU()])
        if scale != 4:
            self.down_sample_2 = nn.Sequential(
                *[nn.Conv2d(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding), nn.PReLU()])
        self.error_encode = nn.Sequential(
            *[nn.ConvTranspose2d(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding),
              nn.PReLU()])
        self.post_conv = BasicBlock(conv, in_channel, in_channel, kernel_size, stride=1, bias=True,
                                           act=nn.PReLU())
        if scale == 4:
            self.multi_source_projection_2 = MultisourceProjection(in_channel, kernel_size=kernel_size, scale=scale,
                                                                   conv=conv)
            self.down_sample_3 = nn.Sequential(*[nn.Conv2d(in_channel, in_channel, 8, stride=4, padding=2), nn.PReLU()])
            self.down_sample_4 = nn.Sequential(*[nn.Conv2d(in_channel, in_channel, 8, stride=4, padding=2), nn.PReLU()])
            self.error_encode_2 = nn.Sequential(
                *[nn.ConvTranspose2d(in_channel, in_channel, 8, stride=4, padding=2), nn.PReLU()])

    def forward(self, x):
        x_up = self.multi_source_projection(x)
        x_down = self.down_sample_1(x_up)
        error_up = self.error_encode(x - x_down)
        h_estimate = x_up + error_up
        if self.scale == 4:
            x_up_2 = self.multi_source_projection_2(h_estimate)
            x_down_2 = self.down_sample_3(x_up_2)
            error_up_2 = self.error_encode_2(x - x_down_2)
            h_estimate = x_up_2 + error_up_2
            x_final = self.post_conv(self.down_sample_4(h_estimate))
        else:
            x_final = self.post_conv(self.down_sample_2(h_estimate))

        return x_final, h_estimate


class CSNLN(nn.Module):
    def __init__(self, n_colors=3, conv=common.default_conv):
        super(CSNLN, self).__init__()

        # n_convblock = args.n_convblocks
        n_feats = 128
        self.depth = 12
        kernel_size = 3
        scale = 4

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)

        # define head module
        m_head = [
            BasicBlock(conv, n_colors, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU()),
            BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU())]

        # define Self-Exemplar Mining Cell
        self.SEM = RecurrentProjection(n_feats, scale=scale)

        # define tail module
        m_tail = [
            nn.Conv2d(
                n_feats * self.depth, n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, input):
        x = self.sub_mean(input)
        x = self.head(x)
        bag = []
        for i in range(self.depth):
            x, h_estimate = self.SEM(x)
            bag.append(h_estimate)
        h_feature = torch.cat(bag, dim=1)
        h_final = self.tail(h_feature)
        return self.add_mean(h_final)

