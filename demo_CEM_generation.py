import os,sys
from PIL import Image
import argparse
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from CEM_Demo.SaliencyModel.utils import cv2_to_pil, pil_to_cv2
from CEM_Demo.demo_ResultTable import CEM_Table
from CEM_Demo.utils import generate_CE_map,plot_heatmap_PSNR,read_config



# Main function
def main(config):
    Imagelist = config['Imagelist']
    Modellist = config['Modellist']
    prefix = config['prefix']
    Coarse_num = config['coarse_num']
    Fine_num = config['fine_num']
    ROI_size = config['ROI_size']
    h = config['h']
    w = config['w']
    mask_block_size = config['mask_block_size']
    sr = config['sr']
    tol = config['tol']
    task = config['task']
    TextImg_path = config['TextImg_path']
    # Further setup
    mask_block_size = mask_block_size // sr
    alpha = 0.1
    result = []

    for Model_name in Modellist:
        for Image_name in Imagelist:
            img_hr = Image.open(os.path.join(TextImg_path, f'{Image_name}')).convert('RGB')
            if task == 'SR':
                img_lr = Image.open(os.path.join(TextImg_path, f'{Image_name}-lr.png')).convert('RGB')
            elif task == 'DN':
                img_lr = Image.open(os.path.join(TextImg_path, f'{Image_name}-noise.png')).convert('RGB')
            elif task == 'DR':
                img_lr = Image.open(os.path.join(TextImg_path, f'{Image_name}-rain.png')).convert('RGB')
            os.makedirs(f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}/', exist_ok=True)

            # Load PSNR data from .mat files
            O = \
            sio.loadmat(f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR-Origin.mat')[
                'PSNR']
            Fine_perturb = sio.loadmat(
                f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-F{Fine_num}.mat')[
                'PSNR']
            Coarse_perturb = sio.loadmat(
                f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-C{Coarse_num}.mat')[
                'PSNR']
            CEM_C = np.mean(Coarse_perturb, 2)
            CEM_F = np.mean(Fine_perturb, 2)
            nonzero_indices = np.nonzero(CEM_F)  # 找到 CEM_F 中非零元素的索引
            for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
                CEM_C[i, j] = CEM_F[i, j]

            # Compute CEM
            CEM = O - CEM_C
            CEM[np.abs(CEM) < tol] = 0
            CEM_a = CEM
            CEM_a[h // (sr * mask_block_size):(h + ROI_size) // (sr * mask_block_size),
            w // (sr * mask_block_size):(w + ROI_size) // (sr * mask_block_size)] = 999

            X = np.zeros(3)
            X[0] = np.sum(CEM_a < 0)
            X[1] = np.sum(CEM_a > 0)
            X[2] = np.sum(CEM_a == 0)
            result.append((Model_name, Image_name, X))

            # Generate causal map
            if task == 'SR':
                bound = 2.5
            elif task == 'DN':
                bound = 1
            elif task == 'DR':
                bound = 2
            CEM, minx, miny, min, maxx, maxy, max = generate_CE_map(CEM, w, h, ROI_size, bound=bound,
                                                                        block_size=mask_block_size, sr=sr)
            print(
                f'Effect range for {Model_name}-{Image_name} is {np.round(max - min, 2)} \n From {np.round(max, 2)}({maxx},{maxy}) to {np.round(min, 2)}({minx},{miny}) dB')

            # Visualize the results
            draw_img = pil_to_cv2(CEM)
            cv2.rectangle(draw_img, (w * sr, h * sr), ((w + ROI_size) * sr, (h + ROI_size) * sr), (0, 255, 0), 2)
            A = pil_to_cv2(CEM)
            A[h:h + ROI_size, w:w + ROI_size, :] = [0, 255, 0]
            blend_abs_and_input = cv2_to_pil(A * (1 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)

            plot_heatmap_PSNR(max_value=max, min_value=min, prefix=prefix, ROI_size=ROI_size,
                              block_size=mask_block_size, Model_name=Model_name, Image_name=Image_name,
                              lower_bound=-bound, upper_bound=bound,task=task)

            draw_img2 = pil_to_cv2(blend_abs_and_input)
            cv2.rectangle(draw_img2, (w * sr, h * sr), ((w + ROI_size) * sr, (h + ROI_size) * sr), (0, 255, 0), 2)
            position_pil2 = cv2_to_pil(draw_img2)
            pa = f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}/CEM_PSNR.png'
            position_pil2.save(pa)

    html_saving_path = f'./{task}_{prefix}_Result.html'
    CEM_Table(TextImg_path=TextImg_path,imglist=Imagelist, model_list=Modellist, prefix=prefix, ROI_size=ROI_size,
                      block_size=mask_block_size, html_saving_path=html_saving_path, npz_data=result,task=task)



if __name__ == "__main__":
    # Create argparse parser
    parser = argparse.ArgumentParser(description="CEM Generation.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    # Read config
    config = read_config(args.config)

    # Run main function with config
    main(config)

