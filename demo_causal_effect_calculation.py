import argparse
import yaml
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch, matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import math,sys
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from CEM_Demo.ModelZoo.utils import Tensor2PIL
from CEM_Demo.ModelZoo import load_model,load_denoise_model,load_derain_model
from CEM_Demo.SaliencyModel.utils import pil_to_cv2, calculate_psnr
from CEM_Demo.utils import Get_Intervention_Dataset, load_psnr_data
from CEM_Demo.utils import RainGenerator,extract_first_number_from_image_name

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.data)  # Use len() instead of .shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        img_tensor = self.transform(img)
        return img_tensor


def main(config):
    # Load the configuration
    Imagelist = config['Imagelist']
    Modellist = config['Modellist']
    prefix = config['prefix']
    patch_path = config['patch_path']
    coarse_num = config['coarse_num']
    fine_num = config['fine_num']
    ROI_size = config['ROI_size']
    h = config['h']
    w = config['w']
    mask_block_size = config['mask_block_size']
    load_previous = config['load_previous']
    tol = config['tol']
    sr = config['sr']
    task = config['task']
    TextImg_path = config['TextImg_path']
    mask_block_size = mask_block_size // sr

    # Iterate over the models and images
    for Model_name in Modellist:
        # Get batch size from the config dictionary, fallback to default if not found
        batch_size = config['batch_size_dict'].get(Model_name, 100)
        # Load and prepare the model
        if task == 'SR':
            model = load_model(f'{Model_name}@Base').eval().cuda()
        elif task == 'DN':
            model = load_denoise_model(f'{Model_name}@Base').eval().cuda()
        elif task == 'DR':
            model = load_derain_model(f'{Model_name}@Base').eval().cuda()
        for Image_name in Imagelist:


            hr_pil = Image.open(os.path.join(TextImg_path, f'{Image_name}')).convert('RGB')
            img = np.asarray(hr_pil)
            sizex, sizey = hr_pil.size
            hr_pil = hr_pil.crop((0, 0, sizex - sizex % sr, sizey - sizey % sr))
            sizex, sizey = hr_pil.size
            pil2tensor = transforms.ToTensor()

            if task == 'SR':
                lr_pil = hr_pil.resize((sizex // sr, sizey // sr), Image.BICUBIC)
                lr_pil.save(os.path.join(TextImg_path, f'{Image_name}-lr.png'))
                # plt.imshow(lr_pil)
                # plt.show()
                lr_origin = pil2tensor(lr_pil)
                beta = None
                rain = None
                input_image = lr_pil
            elif task == 'DN':
                img = np.array(img / 255, dtype=float)
                mean = 0
                sigma = 50 / 255
                np.random.seed(0)
                gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
                gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2])
                noise_img = np.clip((gauss + img), 0, 1)
                noise_img = np.uint8(noise_img * 255)
                img = np.array(img * 255, dtype=float)
                lr_pil = Image.fromarray(noise_img)
                lr_pil.save(os.path.join(TextImg_path, f'{Image_name}-noise.png'))
                lr_origin = pil2tensor(lr_pil)
                beta = None
                rain = None
                input_image = lr_pil

            elif task == 'DR':
                first_number = extract_first_number_from_image_name(Image_name)
                RG = RainGenerator(first_number)
                beta, rain, img_rain = RG(img)
                lr_pil = Image.fromarray(img_rain)
                # plt.imshow(lr_pil)
                # plt.show()
                lr_pil.save(os.path.join(TextImg_path, f'{Image_name}-rain.png'))
                lr_origin = pil2tensor(lr_pil)
                input_image = hr_pil

            lr_origin = lr_origin.unsqueeze(0).cuda()
            num_blocks_size = math.ceil(lr_pil.size[0] / mask_block_size)
            sr_origin = model(lr_origin)
            sr_for_retangle = sr_origin.detach().clone()
            sr_for_retangle = Tensor2PIL(torch.clamp(sr_for_retangle, min=0., max=1.))
            sr_origin = sr_origin.squeeze(0).detach().cpu().numpy()
            sr_origin = np.transpose(sr_origin, (1, 2, 0)).clip(0, 1)
            sr_origin = np.uint8(np.round(sr_origin * 255))

            # Define the base output folder
            output_folder = os.path.join(f'./{task}-CEM', f'{prefix}-R{ROI_size}M{mask_block_size}', Model_name)
            os.makedirs(output_folder, exist_ok=True)

            # Define the image output paths
            origin_output_path = os.path.join(output_folder, f"{task}_{Image_name}-Origin.png")
            rectangle_folder = os.path.join(output_folder, Image_name)
            rectangle_output_file = os.path.join(rectangle_folder, "rectangle-result-origin.png")

            # Save the original SR image
            cv2.imwrite(origin_output_path, cv2.cvtColor(sr_origin, cv2.COLOR_BGR2RGB))

            # Draw a rectangle on the SR image
            draw_img = pil_to_cv2(sr_for_retangle)
            cv2.rectangle(draw_img, (w, h), (w + ROI_size, h + ROI_size), (0, 255, 0), 2)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

            # Show the image with the rectangle
            # plt.imshow(draw_img)
            # plt.show()

            # Save the image with the rectangle in the subfolder
            os.makedirs(rectangle_folder, exist_ok=True)
            cv2.imwrite(rectangle_output_file, cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))

            GT_ROI = img[h: h + ROI_size, w: w + ROI_size, :]
            sr_origin_ROI = sr_origin[h: h + ROI_size, w: w + ROI_size, :]

            psnr_origin = calculate_psnr(sr_origin_ROI, GT_ROI)
            sio.savemat(
                f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR-Origin.mat',
                {'PSNR': psnr_origin})

            # load previous data
            PSNR_C, PSNR_F = load_psnr_data(prefix, ROI_size, mask_block_size, Model_name, Image_name,
                                            coarse_num, fine_num,
                                            load_previous,num_blocks_size,task)
            coarse_coordinates_dict = {}
            fine_coordinates_dict = {}

            for i in range(num_blocks_size):
                for j in range(num_blocks_size):
                    if PSNR_C[i, j, -1] != 0 and PSNR_C[i, j, -2] != 0:
                        print(f'{Model_name}-block {i}-{j} coarse is processed')
                        continue
                    for k in range(coarse_num):
                        coarse_coordinates_dict[len(coarse_coordinates_dict)] = (i, j)

            Coarse_Interventions = Get_Intervention_Dataset(coarse_coordinates_dict, input_image, mask_block_size,
                                                               patch_path, task = task,beta=beta,rain=rain)
            Coarse_dataset = CustomDataset(Coarse_Interventions)
            dataloader = DataLoader(Coarse_dataset, batch_size=batch_size, shuffle=False)

            ith_batch = 0
            with torch.no_grad():
                for lr in tqdm(dataloader):
                    print(f'Coarse stage: {patch_path}  {Image_name}/{Model_name}')
                    sr_batch = model(lr.cuda())
                    def process_coarse_image(i):
                        x, y = coarse_coordinates_dict[ith_batch * batch_size + i]
                        sr_one = sr_batch[i].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                        sr_one = np.uint8(np.round(sr_one * 255))
                        sr_ROI = sr_one[h: h + ROI_size, w: w + ROI_size, :]
                        mini_ind = (ith_batch * batch_size + i) % coarse_num
                        PSNR_C[x, y, mini_ind] = calculate_psnr(sr_ROI, GT_ROI)

                    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以调整max_workers的值
                        futures = [executor.submit(process_coarse_image, i) for i in range(sr_batch.shape[0])]
                    for future in futures:
                        future.result()
                    sio.savemat(
                        f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-C{coarse_num}.mat',
                        {'PSNR': PSNR_C})
                    ith_batch += 1
            torch.cuda.empty_cache()

            PSNR_error_C = np.abs(psnr_origin - PSNR_C)
            for cx in range(num_blocks_size):
                for cy in range(num_blocks_size):
                    if PSNR_F[cx, cy, -1] != 0 and PSNR_F[cx, cy, -2] != 0:
                        print(f'{Model_name}-block {cx}-{cy} fine is processed')
                        continue
                    elif np.max(PSNR_error_C[cx, cy, :]) >= tol:
                        for m in range(fine_num):
                            fine_coordinates_dict[len(fine_coordinates_dict)] = (cx, cy)
            #
            Fine_Interventions = Get_Intervention_Dataset(fine_coordinates_dict, input_image, mask_block_size, patch_path,task=task,beta=beta,rain=rain)
            Fine_dataset = CustomDataset(Fine_Interventions)
            dataloader = DataLoader(Fine_dataset, batch_size=batch_size, shuffle=False)
            ith_batch = 0
            with torch.no_grad():
                for lr in tqdm(dataloader):
                    print(f'Fine stage: {patch_path}  {Image_name}/{Model_name}')
                    sr_batch = model(lr.cuda())

                    def process_fine_image(i):
                        x, y = fine_coordinates_dict[ith_batch * batch_size + i]
                        sr_one = sr_batch[i].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                        sr_one = np.uint8(np.round(sr_one * 255))
                        sr_ROI = sr_one[h: h + ROI_size, w: w + ROI_size, :]
                        mini_ind = (ith_batch * batch_size + i) % fine_num
                        PSNR_F[x, y, mini_ind] = calculate_psnr(sr_ROI, GT_ROI)

                    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以调整max_workers的值
                        futures = [executor.submit(process_fine_image, i) for i in range(sr_batch.shape[0])]
                    for future in futures:
                        future.result()
                    ith_batch += 1
                    sio.savemat(
                        f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-F{fine_num}.mat',
                        {'PSNR': PSNR_F})
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEM Configuration")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    main(config)
