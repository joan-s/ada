import pickle
from anytree.walker import Walker
import numpy as np
import random
from tqdm import tqdm
from anytree import LevelOrderIter


with open('results/sort by maxlogit i pseudos lca 10k 3 rounds/tree.pkl', 'rb') as f:
    root = pickle.load(f)
    nodes = pickle.load(f)

leaves = [n for n in nodes if n.is_leaf]
for leaf in leaves:
    leaf.gt = np.argmax(leaf.counts_gt_pixels)


num_annotations = 64000
num_leaves = len(leaves)
print('num leaves', num_leaves, 'num annotations', num_annotations)
# random anotation
#annotated_leaves = random.sample(leaves, k=num_annotations)
# more uncertain / farthest away from nearest class prototype
sorted_leaves = sorted(leaves, key=lambda leaf: leaf.max_logit)
#annotated_leaves = sorted_leaves[(num_leaves-num_annotations)//2:(num_leaves+num_annotations)//2]
#annotated_leaves = sorted_leaves[-num_annotations:]
annotated_leaves = sorted_leaves[:num_annotations]

for leaf in annotated_leaves:
    leaf.annotation = leaf.gt

# propagate to nearest leaves according to max dist to LCA
max_lca_dist = 2
print('max_lca_dist', max_lca_dist)
num_classes = 19
num_masks_per_class = dict()
num_pixels_per_class = dict()
num_mask_errors_per_class = dict()
num_pixel_errors_per_class = dict()
for c in range(num_classes):
    num_masks_per_class[c] = 0
    num_pixels_per_class[c] = 0
    num_mask_errors_per_class[c] = 0
    num_pixel_errors_per_class[c] = 0

total_pseudos = 0
for i,leaf in enumerate(annotated_leaves):
    common_ancestor = leaf
    #print('i',i, 'leaf', leaf.id)
    for j in range(1, max_lca_dist):
        #print('j',j)
        if not common_ancestor.is_root:
            common_ancestor = common_ancestor.parent
            #print('j', j, 'common_ancestor', common_ancestor.id)
            descendants = LevelOrderIter(common_ancestor, maxlevel=max_lca_dist-j+1)
            for desc in descendants:
                if desc.is_leaf and not hasattr(desc, 'annotation'):
                    desc.annotation = leaf.annotation
                    #print('added leaf', desc.id)
                    total_pseudos += 1
                    num_masks_per_class[desc.annotation] += 1
                    num_pixels_per_class[desc.annotation] += desc.area
                    if desc.annotation != desc.gt:
                        num_mask_errors_per_class[desc.gt] += 1
                        num_pixel_errors_per_class[desc.gt] += desc.area

print(total_pseudos, 'total_pseudos')
print('num_masks_per_class', num_masks_per_class)
print('num_mask_errors_per_class', num_mask_errors_per_class)
print('% error masks')
for c in range(num_classes):
    if num_masks_per_class[c]>0:
        err_mask = np.round(100*num_mask_errors_per_class[c]/num_masks_per_class[c], decimals=2)
        print('class', c, err_mask, '%')
    else:
        print('class', c, 'has no annotated masks')
print('num_pixels_per_class', num_pixels_per_class)
print('num_pixel_errors_per_class', num_pixel_errors_per_class)
print('% error pixels')
for c in range(num_classes):
    if num_pixels_per_class[c] > 0:
        err_pix = np.round(100*num_pixel_errors_per_class[c]/num_pixels_per_class[c], decimals=2)
        print('class', c, err_pix, '%')
    else:
        print('class', c, 'has no annotated pixels')

"""
for leaf in tqdm(annotated_leaves):
    common_ancestor = leaf
    leaves_to_annotate = []
    for i in range(max_lca_dist):
        common_ancestor = common_ancestor.parent
        descendant_leaves = common_ancestor.leaves
        max_leaf_depth = common_ancestor.depth + max_lca_dist - i
        for desc_leaf in descendant_leaves:
            if desc_leaf.depth <= max_leaf_depth\
                    and not hasattr(desc_leaf, 'annotation'):
                leaves_to_annotate.append(desc_leaf)
        for l in leaves_to_annotate:
            if not hasattr(l, 'annotation'):
                l.annotation = leaf.gt
                if l.gt != l.annotation:
                    num_mask_errors_per_class[l.gt] += 1
                    num_pixel_errors_per_class[l.gt] += l.area

    #break
"""











