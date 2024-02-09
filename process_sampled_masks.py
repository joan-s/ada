"""
computes inference of those images to extract the mean feature, mean logit and counts predictions of their
sampled masks for a certain network config file and checkpoint. Saves the list of masks (a list of a dictionaries)
into a pkl file.
No faig servir el config de path-checkpoint perque es d'una versio anterior de openmmlab, de Jose Luis
"""

import os
import numpy as np
import pickle
import cv2
from mmseg.apis import inference_model, init_model
from time import time
import argparse
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# SCALE = 64
DOWNSAMPLE = 4 # see compute_features_one_image()


def read_masks_segmentation(fname, dir_masks):
    fnmasks = os.path.join(dir_masks, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


def read_image(fname, dir_images):
    fnima = os.path.join(dir_images, fname + '_leftImg8bit.png')
    ima = cv2.imread(fnima)
    # abans era aixo cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB), no sera que no calia fer la conversio ??
    return ima


def compute_features_one_image(fn, model, dir_images, dir_masks):
    #print(fn)
    img = read_image(fn, dir_images)
    result = inference_model(model, img)

    #probs = softmax(SCALE*result.seg_logits.data, dim=0)
    #probs = probs.cpu().numpy()
    #probs = np.transpose(probs, [1,2,0])

    # logits = model.decode_head.cos_sim[0]
    logits = result.seg_logits.data # [19, 1024, 2048] this is the cosine similarity
    # there is a resize() in the middle, done in inference_model = bilinear interpolation of cos_sim in head
    #logits = logits[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
    logits = logits.cpu().numpy()
    logits = np.transpose(logits, [1, 2, 0])
    #print('logits', logits.shape)

    # pred = np.argmax(logits, axis=2)
    pred = result.pred_sem_seg.data[0] # [1024, 2048]
    # this version of the prediction is better than argmax model.decode_head.cos_sim[0] because resize() =>
    # bilinear interpolation that smoothes prediction, removing noise
    #pred = pred[::DOWNSAMPLE, ::DOWNSAMPLE]
    pred = pred.cpu().numpy()
    #print('pred', pred.shape)

    # feat = model.decode_head.normalized_features([1024, 2048])
    # does resize=bilinear interpolation to ori_shape, so 256 x 1024 x 2048, too large to save for CS train
    feat_downsampled = model.decode_head.feat_normalized[0]
    # no resize(), so [256, 256, 512], features in first dimension => DOWNSAMPLE = 4
    feat_downsampled = feat_downsampled.cpu().numpy()
    feat_downsampled = np.transpose(feat_downsampled, [1, 2, 0]) # [256, 512, 256], features in last dimension
    #print('feat_downsampled', feat_downsampled.shape)

    assert np.all(~np.isnan(pred))
    assert np.all(~np.isnan(feat_downsampled))
    assert np.all(~np.isnan(logits))
    return logits, pred, feat_downsampled


def process_masks_one_image(fn, logits, pred, feat_downsampled, masks_in_image, dir_masks, num_classes):
    masks_segmentation = read_masks_segmentation(fn, dir_masks)
    for m in masks_in_image:
        seg = masks_segmentation[m['nmask_in_image']]['segmentation']
        seg_downsampled = seg[::DOWNSAMPLE,::DOWNSAMPLE]
        m['mean_feature'] = feat_downsampled[seg_downsampled].mean(axis=0)
        # m['mean_feature'] = np.linalg.norm(feat[seg_downsampled].mean(axis=0))
        # the mean of normalized vectors is not normalized
        # downsample because the features are donwsampled, see compute_features_one_image()
        m['mean_logit'] = logits[seg].mean(axis=0)
        m['counts_pred'] = np.bincount(pred[seg], minlength=num_classes)



def compute_features_and_process_masks_one_image(model, fn, masks_fn, show_progress, dir_images, dir_masks,
                                                 num_classes):
    logits, pred, feat = compute_features_one_image(fn, model, dir_images, dir_masks)
    process_masks_one_image(fn, logits, pred, feat, masks_fn, dir_masks, num_classes)
    if show_progress:
        print('.', end='', flush=True)


def compute_features_and_process_masks(config, checkpoint, dict_masks, sequential, dir_images, dir_masks,
                                       num_classes):
    model = init_model(config, checkpoint, 'cuda:0')
    model.cfg.model.test_cfg.mode = 'whole'
    fnames = list(dict_masks.keys())
    if sequential:
        show_progress = False
        for fn in tqdm(fnames):
            compute_features_and_process_masks_one_image(model, fn, dict_masks[fn], show_progress, dir_images,
                                                         dir_masks, num_classes)
    else:
        import torch
        torch.multiprocessing.set_start_method('spawn')
        show_progress = True
        with ThreadPool(processes=2) as pool:
            # the limitation of processes is to avoid CUDA errors
            pool.starmap(
                compute_features_and_process_masks_one_image,
                [(model, fn, dict_masks[fn], show_progress, dir_images, dir_masks, num_classes)
                 for i,fn in enumerate(fnames)]
            )


def load_sampled_masks(fname_sampled_masks):
    with open(fname_sampled_masks, 'rb') as f:
        dict_masks_sampled = pickle.load(f)

    print('loaded dict_masks_sampled from {}'.format(fname_sampled_masks))
    return dict_masks_sampled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-sampled-masks', required=True)
    parser.add_argument('--path-checkpoint', required=True)
    parser.add_argument('--path-config', required=True)
    parser.add_argument('--dir-masks', required=True)
    parser.add_argument('--dir-images', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--path-out', required=True)
    parser.add_argument('--sequential', action="store_true", required=False)
    parser.set_defaults(sequential=False)
    return parser.parse_args()


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # doesn't work in pycharm terminals

    args = parse_args()
    if os.path.exists(args.path_out):
        print('file {} already exists, exiting script process_sampled_masks.py'.format(args.path_out))
        exit()

    dict_masks_sampled = load_sampled_masks(args.path_sampled_masks)
    t0 = time()
    # compute_features_and_process_masks_sequential(args.path_config, args.path_checkpoint, dict_masks_sampled)
    compute_features_and_process_masks(args.path_config, args.path_checkpoint, dict_masks_sampled,
                                       args.sequential, args.dir_images, args.dir_masks, args.num_classes)
    t1 = time()
    print('time {} s.'.format(t1 - t0))
    with open(args.path_out, 'wb') as f:
        pickle.dump(dict_masks_sampled, f)

    print('\nsaved dict of processed masks to {}'.format(args.path_out))