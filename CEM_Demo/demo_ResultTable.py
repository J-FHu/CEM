import cv2
import os
from CEM_Demo.Training.utils.html_viz import get_sortable_html_header,get_sortable_html_footer, convert_image_to_html

#%%
def find_matrix(results, M, I):
    for data in results:
        A, B, D = data
        if A == M and B == I:
            return D
    return None


def CEM_Table(TextImg_path,imglist, model_list, prefix,ROI_size,block_size,html_saving_path,npz_data,task):


    # model_list = sorted(os.listdir(exp_path))
    img_list = [file for file in imglist]
    img_list = sorted(img_list)
    Image_size = 256

    if task == 'SR':
        COL_NAME_LIST = ['SR']
    elif task == 'DN':
        COL_NAME_LIST = ['DN']
    elif task == 'DR':
        COL_NAME_LIST = ['DR']
    for imgname in img_list:
        COL_NAME_LIST = COL_NAME_LIST + [imgname, imgname]
    header = get_sortable_html_header(COL_NAME_LIST)
    footer = get_sortable_html_footer()
    html = ''

    html += '<tr>\n'
    html += '  <td>' + f'Index' + '</td>\n'
    for i, img_name in enumerate(img_list):
        html += f'<td>{i}</td><td></td>\n'
    html += '</tr>\n'

    html += '<tr>\n'
    html += '  <td>' + f'GT and LR images' + '</td>\n'
    for img_name in img_list:
        img_cv2 = cv2.imread(os.path.join(TextImg_path, f'{img_name}'))
        if task == 'SR':
            img_cv2_lr = cv2.imread(os.path.join(TextImg_path, f'{img_name}-lr.png'))
        elif task == 'DN':
            img_cv2_lr = cv2.imread(os.path.join(TextImg_path, f'{img_name}-noise.png'))
        elif task == 'DR':
            img_cv2_lr = cv2.imread(os.path.join(TextImg_path, f'{img_name}-rain.png'))
        try:
            img_cv2 = cv2.resize(img_cv2, (Image_size, Image_size))
            img_cv2_lr = cv2.resize(img_cv2_lr, (Image_size, Image_size))
            html += '  <td>' + convert_image_to_html(img_cv2) + '</td><td>' + convert_image_to_html(img_cv2_lr) + '</td>\n'
        except:
            html += '  <td></td><td></td>\n'
    html += '</tr>\n'


    for i, model_name in enumerate(model_list):

        iii_path = f'./{task}-CEM/{prefix}-R{ROI_size}M{block_size}/{model_name}/'
########################
        html += '<tr>\n'
        html += '  <td>' + f'{model_name} Result and CEM' + '</td>\n'
        for j, img_name in enumerate(img_list):
            m = os.path.join(iii_path, f'{img_name}/rectangle-result-origin.png')
            img_cv2 = cv2.imread(m)
            # plt.imshow(img_cv2)
            # plt.show()
            try:
                img_cv2 = cv2.resize(img_cv2, (Image_size, Image_size))
                html += '  <td>' + convert_image_to_html(img_cv2) + '</td>\n'
            except:
                html += '  <td></td>\n'

            img_cv2 = cv2.imread(os.path.join(iii_path, f'{img_name}/CEM_PSNR.png'))
            try:
                img_cv2 = cv2.resize(img_cv2, (Image_size, Image_size))
                html += '  <td>' + convert_image_to_html(img_cv2) + '</td>\n'
            except:
                html += '  <td></td>\n'
        html += '</tr>\n'

######################

        html += '<tr>\n'
        html += '  <td>' + f'{model_name}: Result and CEM' + '</td>\n'
        for j, img_name in enumerate(img_list):

            html += f'  <td> {model_name}: Result </td>\n'

            html += f'  <td> {model_name}: CEM </td>\n'

        html += '</tr>\n'
        # plt.savefig(f'./SR-CEM/{prefix}-R{ROI_size}M{block_size}/{Model_name}/{Image_name}/Range_PSNR.png')
######################
        html += '<tr>\n'
        html += '  <td>' + f'{model_name}: Effect patch number and range' + '</td>\n'
        for j, img_name in enumerate(img_list):
            D = find_matrix(npz_data, model_name, img_name)
            try:
                html += '  <td>' + f'Negative:{D[0]}, Positive:{D[1]}, Zero:{D[2]}' + '</td>\n'
            except:
                html += '  <td></td>\n'

            img_cv2 = cv2.imread(os.path.join(f'./{task}-CEM/{prefix}-R{ROI_size}M{block_size}/{model_name}/{img_name}', f'Range_PSNR.png'))
            try:
                # img_cv2 = cv2.resize(img_cv2, (Image_size, Image_size))
                html += '  <td>' + convert_image_to_html(img_cv2) + '</td>\n'
            except:
                html += '  <td></td>\n'
        html += '</tr>\n'
    with open(html_saving_path, 'w') as f:
        f.write(header + html + footer)
