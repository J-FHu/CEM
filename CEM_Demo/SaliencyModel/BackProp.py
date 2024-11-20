import numpy as np
import torch
import cv2, os, matplotlib.pyplot as plt

from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel, prepare_images, prepare_real_images
from natsort import natsorted
from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one, _remove_batch
import asyncio
def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)
    return calculate_objective

def saliency_map_gradient(numpy_image, model, attr_func):
    img_tensor = torch.from_numpy(numpy_image)
    img_tensor.requires_grad_(True)
    if img_tensor.dim() == 3:
        result = model(_add_batch_one(img_tensor))
    elif img_tensor.dim() == 4:
        result = model(img_tensor)
    target = attr_func(result)
    target.backward()
    return img_tensor.grad.numpy(), result

def I_gradient(numpy_image, baseline_image, model, attr_objective, fold, interp='linear'):
    interpolated = interpolation(numpy_image, baseline_image, fold, mode=interp).astype(np.float32)
    grad_list = np.zeros_like(interpolated, dtype=np.float32)
    result_list = []
    for i in range(fold):
        img_tensor = torch.from_numpy(interpolated[i])
        img_tensor.requires_grad_(True)
        result = model(_add_batch_one(img_tensor))
        target = attr_objective(result)
        target.backward()
        grad = img_tensor.grad.numpy()
        grad_list[i] = grad
        result_list.append(result)
    results_numpy = np.asarray(result_list)
    return grad_list, results_numpy, interpolated


def GaussianBlurPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        h, w, c = cv_numpy_image.shape
        kernel_interpolation = np.zeros((fold + 1, l, l))
        image_interpolation = np.zeros((fold, h, w, c))
        lambda_derivative_interpolation = np.zeros((fold, h, w, c))
        sigma_interpolation = np.linspace(sigma, 0, fold + 1)
        for i in range(fold + 1):
            kernel_interpolation[i] = isotropic_gaussian_kernel(l, sigma_interpolation[i])
        for i in range(fold):
            image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
            lambda_derivative_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, (kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
    return path_interpolation_func


def GaussianLinearPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        kernel = isotropic_gaussian_kernel(l, sigma)
        baseline_image = cv2.filter2D(cv_numpy_image, -1, kernel)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold, axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
    return path_interpolation_func


def LinearPath(fold):
    def path_interpolation_func(cv_numpy_image):
        baseline_image = np.zeros_like(cv_numpy_image)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold, axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
    return path_interpolation_func


def Path_gradient(numpy_image, model, attr_objective, path_interpolation_func, cuda=False):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    if cuda:
        model = model.cuda()

    cv_numpy_image = np.moveaxis(numpy_image, 0, 2)
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in range(image_interpolation.shape[0]):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)
        if cuda:
            # print(img_tensor.shape)
            result = model(_add_batch_one(img_tensor).cuda())
            # print(result.shape)
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        else:
            result = model(_add_batch_one(img_tensor))
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        result=result.clip(0,1)
        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result)
    results_numpy = np.asarray([t.cpu().detach().numpy() for t in result_list])
    return grad_accumulate_list, results_numpy, image_interpolation


def batch_output(batch_size, image_folder_path, attr_objective, model, start=0):
    input_tensors = []
    image_file_names = natsorted(os.listdir(image_folder_path))

    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        if "block" not in image_file_name:
            continue
        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_images(input_image_path)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))

    # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(True)

            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            output_tensor.requires_grad_(True)
            target = attr_objective(output_tensor)
            target.backward()
            grad = batch_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
            break

    return torch.clamp(output_tensor,0,1), grad

def batch_randlib_nolam_output(batch_size, image_folder_path, attr_objective, model, start):
    input_tensors = []
    name_list = []
    image_file_names = natsorted(os.listdir(image_folder_path))
    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_images(input_image_path)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))
        name_list.append(image_file_name)
        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(False)
            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            # target = attr_objective(output_tensor)
            # target.backward()
            break


    return output_tensor,  name_list
def batch_randlib_output(batch_size, image_folder_path, attr_objective, model, start):
    print("start")
    input_tensors = []
    name_list = []
    image_file_names = natsorted(os.listdir(image_folder_path))
    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_images(input_image_path)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))
        name_list.append(image_file_name)
        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(True)
            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            output_tensor.requires_grad_(True)
            target = attr_objective(output_tensor)
            target.backward()
            grad = batch_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
            break

    return output_tensor, grad, name_list


def batch_spixel_random_output(batch_size, image_folder_path, attr_objective, model, spno, start=0):
    input_tensors = []
    name_list = []
    image_file_names = natsorted(os.listdir(image_folder_path))
    for i, image_file_name in enumerate(image_file_names):
        if not image_file_name.startswith(str(spno)):
            continue
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_images(input_image_path)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))
        name_list.append(image_file_name)
        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(True)
            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            output_tensor.requires_grad_(True)
            target = attr_objective(output_tensor)
            target.backward()
            grad = batch_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

    return output_tensor, grad, name_list

def batch_spixel_pooling_output(batch_size, image_folder_path, attr_objective, model, start=0):
    input_tensors = []
    name_list = []
    image_file_names = natsorted(os.listdir(image_folder_path))
    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_images(input_image_path)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))
        name_list.append(image_file_name)
        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(True)
            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            output_tensor.requires_grad_(True)
            target = attr_objective(output_tensor)
            target.backward()
            grad = batch_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

    return output_tensor, grad, name_list

def batch_real_output(gauss,batch_size, image_folder_path, attr_objective, model, start=0):
    input_tensors = []
    output_tensors = []
    image_file_names = natsorted(os.listdir(image_folder_path))

    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue
        if not image_file_name.endswith('.png'):
            continue
        if "block" not in image_file_name:
            continue

        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_lr, img_hr = prepare_real_images(input_image_path,gauss)
        tensor_lr = PIL2Tensor(img_lr)[:3]
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(torch.from_numpy(tensor_lr.numpy()))

        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)
            batch_tensor.requires_grad_(True)

            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
            output_tensor.requires_grad_(True)
            target = attr_objective(output_tensor)
            target.backward()
            grad = batch_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

    return output_tensor, grad

def batch_denoise_output(batch_size, image_folder_path, model, start=0):
    input_tensors = []
    output_tensors = []
    image_file_names = natsorted(os.listdir(image_folder_path))

    for i, image_file_name in enumerate(image_file_names):
        if i < start:
            continue

        # 加载输入图像并进行变换
        input_image_path = os.path.join(image_folder_path, image_file_name)
        img_noise = cv2.imread(input_image_path)
        img_noise = cv2.cvtColor(img_noise, cv2.COLOR_RGB2BGR)
        # plt.imshow(img_noise)
        # plt.show()
        img_noise = np.transpose(img_noise,(2,0,1))
        img_noise = torch.div(torch.from_numpy(img_noise).float(), 255.0)
        # input_tensor = cv2.imread(input_image_path)
        # 添加到批次张量中
        input_tensors.append(img_noise)
        # 如果批次张量已满，则进行一次推理，并将输出张量添加到列表中
        if len(input_tensors) == batch_size:
            # 将批次张量拼接为4维张量
            batch_tensor = torch.stack(input_tensors, dim=0)

            # 推理并得到输出张量
            output_tensor = model(batch_tensor.cuda())
    return output_tensor


def Path_gradient_wo_model(numpy_image, result, grad, path_interpolation_func, cuda=False):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    cv_numpy_image = np.moveaxis(numpy_image, 0, 2)
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in range(image_interpolation.shape[0]):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)

        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.unsqueeze(0))
    results_numpy = np.asarray([t.cpu().detach().numpy() for t in result_list])
    return grad_accumulate_list, results_numpy, image_interpolation

def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


def saliency_map_P_gradient(
        numpy_image, model, attr_objective, path_interpolation_func):
    grad_list, result_list, _ = Path_gradient(numpy_image, model, attr_objective, path_interpolation_func)
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


def saliency_map_I_gradient(
        numpy_image, model, attr_objective, baseline='gaus', fold=10, interp='linear'):
    """
    :param numpy_image: RGB C x H x W
    :param model:
    :param attr_func:
    :param h:
    :param w:
    :param window:
    :param baseline:
    :return:
    """
    numpy_baseline = np.moveaxis(IG_baseline(np.moveaxis(numpy_image, 0, 2) * 255., mode=baseline) / 255., 2, 0)
    grad_list, result_list, _ = I_gradient(numpy_image, numpy_baseline, model, attr_objective, fold, interp='linear')
    final_grad = grad_list.mean(axis=0) * (numpy_image - numpy_baseline)
    return final_grad, result_list[-1]

