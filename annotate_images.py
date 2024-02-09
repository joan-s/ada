import argparse
import os
import time
from itertools import chain

import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from itertools import groupby
from tqdm import tqdm
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import random   # for a different seed in each multiprocessing thread

PALETTE_CS = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
              [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
              [107, 142, 35], [152, 251, 152], [70, 130, 180],
              [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
              [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
              [255, 255, 255]]


def apply_palette(ima, palette=PALETTE_CS):
    assert ima.ndim == 2
    labels = np.unique(ima)
    res = np.zeros((ima.shape[0], ima.shape[1], 3), dtype=np.uint8)
    for lab in labels:
        if lab in range(len(palette)):
            res[ima == lab, :] = palette[lab]  # 19 is mapped to white

    # passed ignore_label (255) is also mapped to white
    res[ima == args.ignore_label] = [255, 255, 255]
    return res


def read_masks(path_masks, fname):
    fnmasks = os.path.join(path_masks, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


def save_annotated(ima, annotated, dir_out_images, dir_out_annotations, fname):
    cv2.imwrite(os.path.join(dir_out_images, fname + '_leftImg8bit.png'), ima)
    cv2.imwrite(os.path.join(dir_out_annotations, fname + '_gtFine_labelTrainIds.png'), annotated)


def annotate_and_save_one_image(fname_image, masks_image, dir_images, dir_masks, dir_out_images,
                                dir_out_annotations, gt_shape, ignore_label):
    annotated_image = np.full(gt_shape, ignore_label, dtype=np.uint8)
    seg_masks_image = read_masks(dir_masks, fname_image)
    num_masks_annotated = 0
    ima = read_image(fname_image, dir_images)
    for m in masks_image:
        idx = m['nmask_in_image']
        seg = seg_masks_image[idx]['segmentation']
        assert not( (m['annotation'] is not None) and (m['pseudolabel'] is not None))
        if m['annotation'] is not None:
            assert m['pseudolabel'] is None
            annotated_image[seg] = m['annotation']
            num_masks_annotated += 1
        elif m['pseudolabel'] is not None:  # m['annotation'] is None
            annotated_image[seg] = m['pseudolabel']
            num_masks_annotated += 1


    save_annotated(ima, annotated_image, dir_out_images, dir_out_annotations, fname_image)
    print('.', end='', flush=True)
    return num_masks_annotated


def save_annotated_images(dict_masks, dir_images, dir_masks, dir_out_images, dir_out_annotations,
                          gt_shape, ignore_label):
    num_cores = cpu_count()
    print('{} cores'.format(num_cores))
    with Pool(processes=num_cores) as pool:
        num_regions_annotated = pool.starmap(annotate_and_save_one_image,
                                             [(fname_image, masks_image, dir_images, dir_masks, dir_out_images,
                                               dir_out_annotations, gt_shape, ignore_label)
                                              for fname_image, masks_image in dict_masks.items()])
    print('\n{} total regions annotated in {} images'
          .format(np.sum(num_regions_annotated), len(dict_masks)))


def get_cities(dict_masks):
    cities = np.unique([fn.split('/')[0] for fn in dict_masks.keys()])
    return cities


def make_dirs(dir_out_images, dir_out_annotations, cities):
    for city in cities:
        new_dir_images = os.path.join(dir_out_images, city)
        assert not os.path.isdir(new_dir_images)
        os.makedirs(new_dir_images)
        new_dir_annotations = os.path.join(dir_out_annotations, city)
        assert not os.path.isdir(new_dir_annotations)
        os.makedirs(new_dir_annotations)


def save_annotation_file(dir_out, suffix, path_ann_file):
    fnames = list(chain.from_iterable(
        [[os.path.join(root, fn) for fn in files] for root, subdirs, files in os.walk(dir_out) if files != []]
    ))
    # remove prefix and suffix
    fnames = [fn.replace(dir_out+'/', '') for fn in fnames]
    fnames = [fn.replace(suffix, '') for fn in fnames]
    with open(path_ann_file, 'w') as f:
        for fn in fnames:
            print(fn, file=f)
    print('saved annotation file {}'.format(path_ann_file))


def read_image(fname, dir_images):
    fnima = os.path.join(dir_images, fname + '_leftImg8bit.png')
    ima = cv2.imread(fnima) # cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
    return ima


def read_masks(path_masks, fname):
    fnmasks = os.path.join(path_masks, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-annotated-masks', required=True)
    parser.add_argument('--dir-images', required=True)
    parser.add_argument('--dir-masks', required=True)
    parser.add_argument('--gt-shape', nargs=2, type=int, required=True)
    parser.add_argument('--ignore-label', type=int, required=True)
    parser.add_argument('--dir-out', required=True)
    parser.add_argument('--dir-out-images', required=True)
    parser.add_argument('--dir-out-annotations', required=True)
    parser.add_argument('--path-ann-file', required=True)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    t1 = time.time()

    with open(args.path_annotated_masks, 'rb') as f:
      dict_annotated_masks = pickle.load(f)

    num_images = len(dict_annotated_masks.keys())
    num_masks = sum([len(v) for v in dict_annotated_masks.values()])
    print('loaded dict of annotated masks from {} : {} images and {} masks'\
        .format(args.path_annotated_masks, num_images, num_masks))

    cities = get_cities(dict_annotated_masks)
    print(cities)
    make_dirs(args.dir_out_images, args.dir_out_annotations, cities)
    save_annotated_images(dict_annotated_masks, args.dir_images, args.dir_masks, args.dir_out_images,
                          args.dir_out_annotations, args.gt_shape, args.ignore_label)
    t2 = time.time()
    print('{} seconds'.format(int(t2-t1)))

    # write .txt annotation file for mmseg training config
    save_annotation_file(args.dir_out_annotations, '_gtFine_labelTrainIds.png', args.path_ann_file)
