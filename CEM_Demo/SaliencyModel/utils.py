import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import cv2
import lpips
import torch
from PIL import ImageFilter
def cv2_to_pil(img):
    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target


def blend_input(map, input):
    return Image.blend(map, input, 0.4)


def count_saliency_pixels(map, threshold=0.95):
    sum_threshold = map.reshape(-1).sum() * threshold
    cum_sum = -np.cumsum(np.sort(-map.reshape(-1)))
    return len(cum_sum[cum_sum < sum_threshold])


def plot_diff_of_attrs_kde(A, B, zoomin=4, blend=0.5):
    grad_flat = A.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:A.shape[0]:1, 0:A.shape[1]:1]
    Y, X = np.mgrid[0:A.shape[0]:1, 0:A.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Za = np.reshape(kernel(positions).T, A.shape)
    Za = Za / Za.max()

    grad_flat = B.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:B.shape[0]:1, 0:B.shape[1]:1]
    Y, X = np.mgrid[0:B.shape[0]:1, 0:B.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Zb = np.reshape(kernel(positions).T, B.shape)
    Zb = Zb / Zb.max()

    diff = Za - Zb
    diff_norm = diff / diff.max()
    vis = Zb - blend*diff_norm

    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(vis * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    grad_flat = grad_flat + 1e-8
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def vis_saliency(map, zoomin=4):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(map)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    Img = Img.resize((s1 * zoomin, s2 * zoomin), Image.NEAREST)
    return Img.convert('RGB')


def click_select_position(pil_img, window_size=16):
    """

    :param pil_img:
    :param window_size:
    :return: w, h
    """
    cv2_img = pil_to_cv2(pil_img)
    position = [-1, -1]
    def mouse(event, x, y, flags, param):
        """"""
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d, %d" % (x, y)
            position[0] = x
            position[1] = y
            draw_img = cv2_img.copy()
            cv2.rectangle(draw_img, (x, y), (x + window_size, y + window_size), (0,0,255), 2)
            cv2.putText(draw_img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness = 1)
            cv2.imshow("image", draw_img)

    cv2.namedWindow("image")
    cv2.imshow("image", cv2_img)
    cv2.resizeWindow("image", 800, 600)
    cv2.setMouseCallback("image", mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return_img = cv2_img.copy()
    cv2.rectangle(return_img, (position[0], position[1]), (position[0] + window_size, position[1] + window_size), (0, 0, 255), 2)
    return position[0], position[1], cv2_to_pil(return_img)


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil

def prepare_real_images(hr_path, gauss, scale=4):
    hr_pil = Image.open(hr_path)
    # RGBA to RGB
    if hr_pil.mode == 'RGBA':
        hr_pil = hr_pil.convert('RGB')
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    hr_img = np.array(hr_pil)
    hr_img = hr_img.astype(np.float32) / 255.0
    # plt.imshow(hr_img)
    # plt.show()
    k_size = (5, 5)

    blur_img = cv2.GaussianBlur(hr_img, ksize=k_size, sigmaX=10)
    # plt.imshow(blur_img)
    # plt.show()

    gauss = gauss.reshape(blur_img.shape[0], blur_img.shape[1], blur_img.shape[2])
    noise_img = np.clip((gauss + blur_img), 0, 1)
    noise_img = np.uint8(noise_img * 255)
    # plt.imshow(noise_img)
    # plt.show()
    noise_pil = Image.fromarray(np.asarray(noise_img, dtype=np.uint8))
    lr_pil = noise_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)


    return lr_pil, hr_pil

def grad_abs_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad.sum(axis=0))
    grad_max = grad_2d.max() + 1e-5
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = grad.sum(axis=0)
    grad_max = max(grad_2d.max(), abs(grad_2d.min()))
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_abs_norm_singlechannel(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad)
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max
    return grad_norm


def IG_baseline(numpy_image, mode='gaus'):
    """
    :param numpy_image: cv2 image
    :param mode:
    :return:
    """
    if mode == 'l1':
        raise NotImplementedError()
    elif mode == 'gaus':
        ablated = cv2.GaussianBlur(numpy_image, (7, 7), 0)
    elif mode == 'bif':
        ablated = cv2.bilateralFilter(numpy_image, 15, 90, 90)
    elif mode == 'mean':
        ablated = cv2.medianBlur(numpy_image, 5)
    else:
        ablated = cv2.GaussianBlur(numpy_image, (7, 7), 0)
    return ablated


def interpolation(x, x_prime, fold, mode='linear'):
    diff = x - x_prime
    l = np.linspace(0, 1, fold).reshape((fold, 1, 1, 1))
    interp_list = l * diff + x_prime
    return interp_list


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def calculate_psnr(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')


# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):

    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def calculate_lpips(img, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)


    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = im2tensor(img[:,:,::-1])
    img2 = im2tensor(img2[:,:,::-1])

    d = loss_fn_alex(img, img2)

    if d == 0:
        return float('inf')
    return float(d)
