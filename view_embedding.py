import os
import numpy as np
from skimage.io import imread
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import cv2
from tqdm import tqdm
from mmseg.apis import inference_model, init_model
import mmcv



CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain',
           'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle')
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180],
           [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
           [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]


def view_embedding_pixels(model, downsample_pixels):
    fnames = [
        'frankfurt/frankfurt_000000_000294',
        'frankfurt/frankfurt_000000_003920',
        'frankfurt/frankfurt_000001_037705',
        'lindau/lindau_000000_000019',
        'lindau/lindau_000014_000019',
        'lindau/lindau_000056_000019',
        'munster/munster_000000_000019',
        'munster/munster_000010_000019',
        'munster/munster_000042_000019',
    ]
    embeddings_pixels = []
    num_samples = []
    for fn in fnames:
        fn_image = 'data/cityscapes/leftImg8bit/val/{}_leftImg8bit.png'.format(fn)
        img = mmcv.imread(fn_image)
        result = inference_model(model, img)
        feat = model.decode_head.normalized_features([1024, 2048])
        emb_pix = feat[0][:, ::downsample_pixels, ::downsample_pixels].cpu().numpy()
        emb_pix = np.transpose(emb_pix, [1, 2, 0])
        emb_pix = np.reshape(emb_pix, [(1024 // downsample_pixels) * (2048 // downsample_pixels), 256])
        embeddings_pixels.append(emb_pix)
        num_samples.append(len(emb_pix))

    embeddings = embeddings_pixels
    print('Computing embedding with t-sne...')
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30., n_jobs=8,
                verbose=1)  # n_iter=1000,
    emb_t = tsne.fit_transform(np.vstack(embeddings))
    print('done')

    num_classes = 19
    num_points_per_embedding = len(emb_t) // len(fnames)
    # markers = ['.', '+', 'o', '^']
    dict_emb_t = {}
    for lab in range(num_classes):
        dict_emb_t[lab] = []
        for i in range(len(fnames)):
            fn_gt = 'data/cityscapes/gtFine_val_8bits/{}_gtFine_labelTrainIds.png'.format(fnames[i])
            img_gt = imread(fn_gt)
            img_gt = img_gt[::downsample_pixels, ::downsample_pixels].flatten()
            idx = img_gt == lab
            dict_emb_t[lab].append(emb_t[i * num_points_per_embedding:(i + 1) * num_points_per_embedding][idx])
        dict_emb_t[lab] = np.vstack(dict_emb_t[lab])
        if len(dict_emb_t[lab]) > 0:
            plt.scatter(dict_emb_t[lab][:, 0], dict_emb_t[lab][:, 1], marker='.',  # markers[lab % len(markers)]
                        label=CLASSES[lab], color=[np.array(PALETTE[lab]) / 255.])
    plt.legend()
    plt.axis('equal')
    plt.show()


def read_masks(path_masks, split, fname):
    fnmasks = os.path.join(path_masks, split, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


def read_image(path_cs, split, fname):
    fnima = os.path.join(path_cs, 'leftImg8bit', split, fname + '_leftImg8bit.png')
    ima = cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
    return ima


def read_gt(path_cs, split, fname):
    path_gt = os.path.join(path_cs, 'gtFine')
    fngt = os.path.join(path_gt, split, fname + '_gtFine_labelTrainIds.png')
    gt = cv2.imread(fngt, cv2.IMREAD_GRAYSCALE)
    return gt


def compute_all_masks_fn(path_cs, path_masks, split, fn, min_area): # 439347 masks in 1/2 hour, min_area=1000
    all_masks_fn = []
    masks = read_masks(path_masks, split, fn)
    gt = read_gt(path_cs, split, fn)
    num_masks_ima = len(masks)
    for nmi in range(num_masks_ima):
        area = masks[nmi]['area']
        if area > min_area:
            seg = masks[nmi]['segmentation']
            majority_label = np.argmax(np.bincount(gt[seg]))
            if majority_label != 255:
                all_masks_fn.append({'fname': fn,
                                  'nmask_in_image': nmi,
                                  'area': area,
                                  'label': majority_label,
                                  'segmentation': seg})
    return all_masks_fn


def view_embedding_regions(model, downsample_filenames):
    path_masks = '../sam/masks_0.86_0.92_400/cityscapes'
    path_cs = 'data/cityscapes/'
    split = 'val'
    min_area = 1000
    # fnames = ['frankfurt/frankfurt_000000_000294', 'frankfurt/frankfurt_000000_000576', ]
    with open(os.path.join(path_cs, 'val.txt'), 'r') as f:
        fnames = f.read().splitlines()
    fnames = fnames[::downsample_filenames] # ::10 500 -> 50 images, 7457 regions

    masks_mean_feature = []
    masks_label = []
    for fn in tqdm(fnames):
        img = read_image(path_cs, split, fn)
        _ = inference_model(model, img)
        feat = model.decode_head.normalized_features([1024, 2048])
        feat = feat[0].cpu().numpy()
        feat = np.transpose(feat, [1,2,0])
        masks = compute_all_masks_fn(path_cs, path_masks, split, fn, min_area)
        # because this is the val split and I've not saved the masks for it
        for m in masks:
            masks_mean_feature.append(feat[m['segmentation']].mean(axis=0))
            masks_label.append(m['label'])

    assert 255 not in masks_label
    masks_mean_feature = np.array(masks_mean_feature)

    embeddings = masks_mean_feature
    print('Computing embedding with t-sne...')
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30., n_jobs=8,
                verbose=1)  # n_iter=1000,
    emb_t = tsne.fit_transform(np.vstack(embeddings))
    print('done')

    markers = ['.', '+',]
    for lab in np.unique(masks_label):
        idx = masks_label==lab
        plt.scatter(emb_t[idx, 0], emb_t[idx, 1], marker=markers[lab % len(markers)],
                    label=CLASSES[lab], color=[np.array(PALETTE[lab]) / 255.])
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint = 'work_dirs/segformer_prototypes/b5_80325/iter_20000.pth'
    config = 'work_dirs/segformer_prototypes/b5_80325/segformer_prototypes_mit-b5_1xb1-20k_80325.py'

    model = init_model(config, checkpoint, device='cuda')
    model.cfg.model.test_cfg.mode = 'whole'

    if False:
        view_embedding_pixels(model, downsample_pixels=16)
    else:
        view_embedding_regions(model, downsample_filenames=10)