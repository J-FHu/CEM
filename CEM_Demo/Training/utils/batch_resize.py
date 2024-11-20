from PIL import Image
import os
import argparse

def batch_resize(args):
    image_list = os.listdir(args.target_images)
    for i, img in enumerate(image_list):
        print(f'Resizing {i}th {img}, ', end='')
        pil = Image.open(
            os.path.join(args.target_images, img)
        )
        w_o, h_o = pil.size
        pil = pil.crop((0, 0, w_o - w_o % args.scale, h_o - h_o % args.scale))
        w, h = pil.size
        print(f'old size {w_o}, {h_o}; new size {w}, {h}')
        pil_lr = pil.resize((int(w / args.scale), int(h / args.scale)), Image.BICUBIC)
        pil_lr.save(os.path.join(args.ouput_path, img))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create lmdb dataset for a image folder.')
    parser.add_argument('-n', '--dataset_name',
                        default='movie_testdemo_lr',
                        help='name of the lmdb dataset')
    parser.add_argument('-i', '--target_images',
                        default='/sdc/jjgu/DF2K',
                        help='Target images to create.')
    parser.add_argument('-o', '--ouput_path',
                        default='/sdc/jjgu/DF2K-LR',
                        help='the output lmdb path')
    parser.add_argument('--scale', type=int, default=4)

    args, other_args = parser.parse_known_args()
    batch_resize(args)
