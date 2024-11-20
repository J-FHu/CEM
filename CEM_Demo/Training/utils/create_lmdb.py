"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""
import argparse
import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2

# from Training.utils.data import ProgressBar
from data import ProgressBar

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def general_image_folder(name, img_folder, lmdb_save_path, BATCH=3000, n_thread=40, read_all_imgs=False):
    """Create lmdb for general image folders
    Users should define the keys, such as: '0321_s035' for DIV2K sub-images
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    #### configurations ####################################
    # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    # After BATCH images, lmdb commits, if read_all_imgs = False
    ########################################################
    meta_info = {'name': name}
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(osp.join(img_folder, '*')))
    keys = []
    for img_path in all_img_list:
        keys.append(osp.splitext(osp.basename(img_path))[0])

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = ProgressBar(len(all_img_list))
        #
        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    resolutions = []
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        txn.put(key_byte, data)
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'vimeo90k':
        key = '00001_0001_4'
    else:
        key = '000_00000000'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create lmdb dataset for a image folder.')
    parser.add_argument('-n', '--dataset_name',
                        default='movie_testdemo_lr',
                        help='name of the lmdb dataset')
    parser.add_argument('-i', '--target_images',
                        default='/sdc/jjgu/DF2K-LR',
                        help='Target images to create.')
    parser.add_argument('-o', '--ouput_path',
                        default='/sdc/jjgu/DF2K-LR.lmdb',
                        help='the output lmdb path')

    print('Intering')
    args, other_args = parser.parse_known_args()
    general_image_folder(args.dataset_name,
                         img_folder=args.target_images,
                         lmdb_save_path=args.ouput_path)