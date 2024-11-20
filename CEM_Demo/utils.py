import cv2
import numpy as np
import os,re
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from CEM_Demo.SaliencyModel.utils import vis_saliency
import yaml
from matplotlib.colors import TwoSlopeNorm



IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm']

def extract_first_number_from_image_name(image_name):
    match = re.search(r'\d+', image_name)
    if match:
        return int(match.group())
    else:
        return None

class RainGenerator:
    rain_cfg_dict = {
        "small": {
            "amount": [0.002, 0.004],
            "width": [1, 3],
            "prob": [0.5, 0.5],
            "length": [25, 35],
            "angle": [-10, 10],
        },
        "medium": {
            "amount": [0.005, 0.007],
            "width": [3, 5, 7],
            "prob": [0.25, 0.5, 0.25],
            "length": [35, 45],
            "angle": [-30, 30],
        },
        "large": {
            "amount": [0.008, 0.010],
            "width": [7, 9],
            "prob": [0.5, 0.5],
            "length": [45, 55],
            "angle": [-60, 60],
        },
        "random": {
            "amount": [0.002, 0.010],
            "width": [1, 3, 5, 7, 9],
            "prob": [0.2, 0.2, 0.2, 0.2, 0.2],
            "length": [25, 55],
            "angle": [-60, 60],
        },
    }

    def __init__(self, seed=None, beta=[0.6, 1.0], rain_types=["medium"], rain_cfg=None):
        self.seed = seed
        self.rain_types = rain_types
        self.rain_cfg = rain_cfg
        self.beta = beta
        self._set_seed()

    def _set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            cv2.setRNGSeed(self.seed)

    def __call__(self, img):
        if self.rain_types:
            assert (
                self.rain_cfg is None
            ), "rain_cfg must be null as rain_types are given"
            rain_type = random.choice(self.rain_types)
        else:
            assert self.rain_cfg, "rain_cfg must be given as rain_types are null"
            rain_type = None
        beta = np.random.uniform(self.beta[0], self.beta[1])
        rain,img_rain = self.add_rain(img, beta, rain_type, self.rain_cfg)
        return beta, rain, img_rain

    def get_noise(self, img, amount):
        """
        生成噪声图像
        >>> 输入:
            img: 图像
            amount: 大小控制雨滴的多少
        >>> 输出:
            图像大小的模糊噪声图像
        """

        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = 256 - amount * 256  # 百分之多少的像素是雨的中心
        noise[np.where(noise < v)] = 0
        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
        noise = cv2.filter2D(noise, -1, k)
        return noise

    def get_rain_blur(self, noise, length=10, angle=0, width=1):
        """
        将噪声加上运动模糊以模仿雨滴
        >>> 输入:
            noise: 输入噪声图, shape = img.shape[0:2]
            length: 对角矩阵大小, 表示雨滴的长度
            angle: 倾斜的角度, 逆时针为正
            width: 雨滴大小
        >>> 输出:
            带模糊的噪声
        """

        # 由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
        trans = cv2.getRotationMatrix2D(
            ((length - 1) / 2, (length - 1) / 2), angle - 45, 1
        )
        dig = np.diag(np.ones(length))  # 生成对焦矩阵
        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (width, width), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

        # k = k / length  # 是否归一化

        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        rain = np.expand_dims(blurred, 2)
        blurred = np.repeat(rain, 3, 2)

        return blurred

    def add_rain(self, img, beta, rain_type, rain_cfg):
        """
        在一张图像中加入雨
        >>> 输入:
            img: 图像
            rain_type: 雨的类型
            beta: 雨的权重
        >>> 输出:
            rain: 雨图
            img_rain: 带雨的图像
        """
        if rain_type:
            assert rain_cfg is None, "rain_cfg must be null as rain_type is given"
            rain_cfg = self.rain_cfg_dict[rain_type]
        amount = rain_cfg["amount"]
        width = rain_cfg["width"]
        prob = rain_cfg["prob"]
        length = rain_cfg["length"]
        # for angle:
        # option 1: [lower, upper]
        # option 2: [[lower1, upper1], [lower2, upper2], ...]
        angle = rain_cfg["angle"]
        if not isinstance(angle[0], int):
            angle = random.choice(angle)

        amount = np.random.uniform(amount[0], amount[1])  # 雨线数目, [0, 1]
        width = np.random.choice(width, p=prob)  # 粗细
        length = np.random.randint(length[0], length[1] + 1)  # 长度
        angle = np.random.randint(angle[0], angle[1] + 1)  # 角度

        noise = self.get_noise(img, amount=amount)
        rain = self.get_rain_blur(noise, length=length, angle=angle, width=width)
        img_rain = cv2.addWeighted(img, 1, rain, beta=beta, gamma=0)

        return rain, img_rain

def get_patch_file(patch_path):
    patch_files = os.listdir(patch_path)
    # print(random.choice(patch_files))
    return random.choice(patch_files), patch_path


def resize_patch(patch, blk_size):
    return cv2.resize(patch, (blk_size, blk_size), interpolation=cv2.INTER_CUBIC)


def Get_Intervention_Dataset(coordinates_dict, input_image, blk_size, patch_path, task, beta =None,rain=None):

    # Initialize variables
    interventions = []


    # Iterate over the coordinates and apply patches
    for key, coord in coordinates_dict.items():
        x, y = coord

        # Select the appropriate patch path based on the counter
        patch_file, patch_path = get_patch_file(patch_path)
        # print(os.path.join(patch_path, patch_file))
        patch = cv2.imread(os.path.join(patch_path, patch_file))

        if task == 'SR':
            patch = resize_patch(patch, blk_size)
            modified_image = np.array(input_image)
            modified_image[x * blk_size:(x + 1) * blk_size, y * blk_size:(y + 1) * blk_size, :] = patch
            # plt.imshow(modified_image)
            # plt.show()
        elif task == 'DN':
            img = np.array(patch / 255, dtype=float)
            mean = 0
            sigma = 50 / 255
            gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
            gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2])
            patch = np.clip((gauss + img), 0, 1)
            patch = np.uint8(patch * 255)
            modified_image = np.array(input_image)
            modified_image[x * blk_size:(x + 1) * blk_size, y * blk_size:(y + 1) * blk_size, :] = patch
            # plt.imshow(modified_image)
            # plt.show()
        elif task == 'DR':
            modified_image = np.array(input_image)
            modified_image[x * blk_size:(x + 1) * blk_size, y * blk_size:(y + 1) * blk_size, :] = patch
            modified_image = cv2.addWeighted(modified_image, 1, rain, beta=beta, gamma=0)
            # plt.imshow(modified_image)
            # plt.show()
        # Append the modified image to the interventions list
        interventions.append(modified_image)


    return interventions


def load_psnr_data(prefix, ROI_size, mask_block_size, model_name, image_name, coarse_num, fine_num, load_previous,
                   num_blocks_size,task):
    # Define the file paths for coarse and fine PSNR data
    coarse_psnr_path = f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{model_name}/{image_name}-PSNR_all_perturb-C{coarse_num}.mat'
    fine_psnr_path = f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{model_name}/{image_name}-PSNR_all_perturb-F{fine_num}.mat'

    # Initialize PSNR arrays with zeros
    PSNR_C = np.zeros((num_blocks_size, num_blocks_size, coarse_num))
    PSNR_F = np.zeros((num_blocks_size, num_blocks_size, fine_num))

    # Load the PSNR data if the corresponding files exist and load_previous is 1
    if load_previous == 1:
        if os.path.exists(fine_psnr_path):
            PSNR_F = sio.loadmat(fine_psnr_path)['PSNR']

        if os.path.exists(coarse_psnr_path):
            PSNR_C = sio.loadmat(coarse_psnr_path)['PSNR']
        elif not os.path.exists(fine_psnr_path):  # If fine PSNR is missing, initialize fine as zeros
            PSNR_F = np.zeros((num_blocks_size, num_blocks_size, fine_num))

    return PSNR_C, PSNR_F

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Function to plot the heatmap for PSNR values
def plot_heatmap_PSNR(max_value, min_value, prefix, ROI_size, block_size, Model_name, Image_name, lower_bound,
                      upper_bound,task):
    # Create a range for heatmap plotting
    array = np.linspace(lower_bound * 2, upper_bound * 2, 2001)
    array[array > max_value] = 0
    array[array < min_value] = 0
    range = np.tile(array, (10, 1))

    if min_value < lower_bound:
        vmin = lower_bound
    else:
        vmin = min_value
    if max_value > upper_bound:
        vmax = upper_bound
    else:
        vmax = max_value
    if min_value < 0 and max_value > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    elif min_value >= 0 and max_value > 0:
        norm = TwoSlopeNorm(vmin=-0.0001, vcenter=vmin, vmax=vmax)
    elif min_value < 0 and max_value <= 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vmax, vmax=0.0001)

    plt.figure(figsize=(2.56, 0.3))
    plt.imshow(range, aspect='auto', cmap='seismic', norm=norm)
    plt.axis('off')
    rows, cols = range.shape
    plt.text(0, rows / 2, f'{min_value:.2f}', fontsize=15, fontweight='bold', ha='center', va='center')
    plt.text(cols, rows / 2, f'{max_value:.2f}', fontsize=15, fontweight='bold', ha='center', va='center')
    plt.savefig(f'./{task}-CEM/{prefix}-R{ROI_size}M{block_size}/{Model_name}/{Image_name}/Range_PSNR.png')


# Custom matrix mapping function
def custom_mapping(matrix, scale):
    result_matrix = np.zeros_like(matrix, dtype=float)
    mapping = (1 / 1.2)
    scale = scale ** mapping
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0:
                result_matrix[i, j] = matrix[i, j] ** mapping
            else:
                result_matrix[i, j] = -(-matrix[i, j]) ** mapping
    return result_matrix / scale



def generate_CE_map(saliency_mat, w, h, ROI_size, bound, block_size=2, sr=4):
    ix = w // sr // block_size
    iy = h // sr // block_size
    ROI_size = ROI_size // sr // block_size
    saliency_mat[iy:iy + ROI_size, ix:ix + ROI_size] = 0

    minx, miny = np.unravel_index(np.argmin(saliency_mat), saliency_mat.shape)
    maxx, maxy = np.unravel_index(np.argmax(saliency_mat), saliency_mat.shape)
    min = np.min(saliency_mat)
    max = np.max(saliency_mat)
    scale = bound
    saliency_mat = saliency_mat.clip(-scale, scale)
    saliency_mat[iy:iy + ROI_size, ix:ix + ROI_size] = scale
    saliency_mat_mapping = custom_mapping(saliency_mat, scale)
    saliency_map = vis_saliency(saliency_mat_mapping, zoomin=block_size * sr)
    return saliency_map, minx, miny, min, maxx, maxy, max