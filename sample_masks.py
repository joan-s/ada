import os
import pickle
from itertools import groupby
import argparse
import random



def sample_dictionary_masks(d, sample_images, sample_masks):
    keys = random.sample(list(d), len(d) // sample_images)
    values = [random.sample(d[k], len(d[k]) // sample_masks) for k in keys]
    return dict(zip(keys, values))


def print_counts(dict_masks):
    num_images = len(dict_masks)
    num_masks = 0
    for k, v in dict_masks.items():
        num_masks += len(v)

    print('sampled masks : {} images, {} masks'.format(num_images, num_masks))


def group_masks_by_image(masks):
    dict_masks = {}
    # all_masks is a list of dictionaries, one dict per mask. Each mask has a
    # fname field. Before we can group all the masks of an image (fname) with groupby
    # we must sort the list by fname
    sorted_masks = sorted(masks, key=(lambda x: x['fname']))
    for k, v in groupby(sorted_masks, key=(lambda x: x['fname'])):
        dict_masks[k] = list(v)
    # key = image name like 'krefeld/krefeld_000000_032845', val = list of its
    # masks, not copied but those in all_masks (a reference)
    return dict_masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-images', type=int, required=True)
    parser.add_argument('--sample-masks', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--path-all-masks', required=True)
    parser.add_argument('--path-out', required=True)
    parser.add_argument('--sequential', action="store_true", required=False)
    parser.set_defaults(sequential=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.path_out):
        print('file {} already exists, exiting script sample_masks.py'.format(args.path_out))
        exit()

    random.seed(args.seed)

    # sample images and masks therein
    with open(args.path_all_masks, 'rb') as f:
        all_masks = pickle.load(f)
    print('loaded {}'.format(args.path_all_masks))

    for mask in all_masks:
        mask['annotation'] = None
        mask['pseudolabel'] = None

    dict_masks = group_masks_by_image(all_masks)
    dict_masks_sampled = sample_dictionary_masks(dict_masks, args.sample_images, args.sample_masks)
    print_counts(dict_masks_sampled)
    with open(args.path_out, 'wb') as f:
        pickle.dump(dict_masks_sampled, f)

    print('\nsaved dict of sampled masks to {}\n'.format(args.path_out))
