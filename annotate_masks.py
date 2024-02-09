import argparse
import os
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from time import time
from sklearn.neighbors import kneighbors_graph
import sys
from anytree import AnyNode, PostOrderIter, PreOrderIter
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from sample_masks import group_masks_by_image
from anytree import LevelOrderIter

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain',
           'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle')


def print_counts(dict_masks):
    num_images = len(dict_masks)
    num_masks = 0
    for k, v in dict_masks.items():
        num_masks += len(v)

    print('sampled masks : {} images, {} masks'.format(num_images, num_masks))


def load_masks(fname_sampled_masks):
    with open(fname_sampled_masks, 'rb') as f:
        dict_masks_sampled = pickle.load(f)

    print('loaded dict_masks_sampled from {}'.format(fname_sampled_masks))
    print_counts(dict_masks_sampled)
    sampled_masks = []
    for v in dict_masks_sampled.values():
        sampled_masks.extend(v)

    return sampled_masks


def make_connectivity_graph(num_neighbors, features):
    print('computing connectivity graph for {} neighbors and {} features...'.format(num_neighbors, features.shape))
    t0 = time()
    knn_graph = kneighbors_graph(features, n_neighbors=num_neighbors, mode='connectivity', metric='euclidean',
                                 include_self=False, n_jobs=-1)
    t1 = time()
    print('time to compute knn_graph {} secs.'.format(t1 - t0))
    print('done')
    return knn_graph


def make_agglomerative_clustering(features, use_connectivity_graph=True, num_neighbors=40):
    # 50 no va
    linkage = 'complete'  # 'single', 'ward', 'average', 'complete'
    metric = 'euclidean'  # because the sem.seg loss was cosine distance to closest groundtruth class prototype
    connectivity = None
    if use_connectivity_graph:
        connectivity = make_connectivity_graph(num_neighbors, features)

    print('computing AgglomerativeClustering...')
    t0 = time()
    hac = AgglomerativeClustering(n_clusters=None, metric=metric,
                                  linkage=linkage, connectivity=connectivity, compute_full_tree=True,
                                  compute_distances=True, distance_threshold=1e12)
    # hac = fastcluster.linkage_vector(features.copy(), method='centroid', metric=metric)
    hac.fit(features)
    t1 = time()
    print('done, {} secs.'.format(t1 - t0))
    return hac


def make_tree(masks, hac):
    num_masks = len(masks)
    sys.setrecursionlimit(max(1000, 2 * num_masks))
    # nodes[80000].leaves
    # RecursionError: maximum recursion depth exceeded while calling a Python object

    children = hac.children_
    nodes = [AnyNode(id=i) for i in range(2 * num_masks - 1)]
    # node.id == mask index in list masks
    for n in nodes:
        n.distance = 0.  # for leaves
    for i in range(len(children)):
        nodes[num_masks + i].children = [nodes[children[i, 0]], nodes[children[i, 1]]]
        nodes[num_masks + i].distance = hac.distances_[i]
    root = nodes[0].root

    print('postorder...')
    t0 = time()
    for node in PostOrderIter(root):
        if node.is_leaf:
            node.mask = masks[node.id]
            node.area = node.mask['area']  # number of pixels of mask
            node.ids_leaves = [node.id]  # node.id is number of mask for this leaf
            node.max_logit = np.max(node.mask['mean_logit'])
            node.counts_gt_pixels = np.array(node.mask['counts_gt'])  # how many pixels have each gt label
            # node.label = np.argmax(node.counts_gt_pixels)  # = masks[node.id]['label']
            # node.counts_gt_masks = np.zeros(num_classes)
            # node.counts_gt_masks[gt] = 1
            node.counts_pred_pixels = node.mask['counts_pred']  # how many pixels classified into each class
            node.pred = np.argmax(node.counts_pred_pixels)
            # node.counts_prediction_masks = np.zeros(num_classes)
            # node.counts_prediction_masks[pred] = 1
            assert (node.mask['annotation'] is not None) + (node.mask['pseudolabel'] is not None) <= 1
            node.annotation = node.mask['annotation']  # either None or not, this is for lca
            node.pseudolabel = node.mask['pseudolabel']  # either None or not, this is for lca
            node.gt = np.argmax(node.counts_gt_pixels)
        else:
            left, right = node.children
            node.area = left.area + right.area
            node.ids_leaves = left.ids_leaves + right.ids_leaves
            node.max_logit = (left.max_logit * left.area + right.max_logit * right.area) / node.area
            node.counts_gt_pixels = left.counts_gt_pixels + right.counts_gt_pixels
            # node.counts_gt_masks = left.counts_gt_masks + right.counts_gt_masks
            # node.counts_prediction_masks = left.counts_prediction_masks + right.counts_prediction_masks
            node.counts_pred_pixels = left.counts_pred_pixels + right.counts_pred_pixels
            node.pred = np.argmax(node.counts_pred_pixels)

    t1 = time()
    print('done, {} s.'.format(t1 - t0))

    return root, nodes


def save_tree(root, nodes):
    with open('./work_dirs/ada/tree.pkl', 'wb') as f:
        pickle.dump(root, f)
        pickle.dump(nodes, f)
    print('saved tree to tree.pkl')


def cluster_features(masks):
    features = np.array([m['mean_feature'] for m in masks])
    # TODO: treure normalitzacio, aixo es pot fer a ada.py.
    #  I encara que no, cal? aixo era per calcular similitud i xe
    norms = np.linalg.norm(features, axis=1)
    features = features / norms[:, np.newaxis]
    # print('\n*** clustering amb logits per comptes de features ***\n')
    # features = np.array([m['mean_logit'].astype(np.float32) for m in masks])
    # features = features[::4]
    # print('\n*** features sampled to {}***\n'.format(features.shape))
    hac = make_agglomerative_clustering(features)
    root, nodes = make_tree(masks, hac)
    save_tree(root, nodes)
    return root, nodes


def select_parents_to_annotate(root, nodes, percent_dists):
    parents = [node for node in nodes if not node.is_leaf]
    distances = [p.distance for p in parents]
    num_bins_distances = 100
    hist_dist, bins_dists = np.histogram(distances, bins=num_bins_distances)
    max_dist = (bins_dists[:-1][100 * np.cumsum(hist_dist) / np.sum(hist_dist) > percent_dists])[1]  # TODO o be [0] ?
    parents_to_annotate = []
    # traverse in pre-order but when selecting a node skip its descendants
    assert not root.is_leaf
    stack = [root]
    while len(stack) > 0:
        node = stack.pop()
        if node.distance < max_dist:
            parents_to_annotate.append(node)
        else:
            for child in node.children:
                if not child.is_leaf:  # only annotate parents=clusters
                    stack.append(child)

    return parents_to_annotate


def compute_pseudolabel(node, num_leaves_to_annotate_per_cluster):
    num_leaves = len(node.ids_leaves)
    num_leaves_to_annotate = min(num_leaves_to_annotate_per_cluster, num_leaves)
    ids_leaves_to_annotate = np.random.choice(node.ids_leaves, num_leaves_to_annotate)
    # les fulles d'un pare es podem fer com node.leaves() o be agafar mask_ids i
    # l'index es l'index a la llista masks i tambe a la llista nodes
    gts_leaves = [np.argmax(nodes[id].counts_gt_pixels) for id in ids_leaves_to_annotate]
    leaves_per_class = np.bincount(gts_leaves)
    pseudolabel = np.argmax(leaves_per_class)
    consensus = leaves_per_class[pseudolabel] / np.sum(leaves_per_class)
    return pseudolabel, consensus, ids_leaves_to_annotate, gts_leaves


def save_annotated_masks(masks, path_out):
    # masks is just a list of masks, we convert it back to a dictionary with
    # key = image filename and value =  list of its masks
    dict_annotated_masks = group_masks_by_image(masks)
    with open(path_out, 'wb') as f:
        pickle.dump(dict_annotated_masks, f)
    print('saved dict of annotated masks to {}'.format(path_out))


def select_random_masks(masks, num_annotations):
    assert len(masks) >= num_annotations
    to_annotate = np.random.choice(masks, size=num_annotations, replace=False)
    for m in to_annotate:
        m['annotation'] = m['label']


def select_masks_sorted_by_maxlogit(masks, num_annotations, num_classes):
    assert len(masks) >= num_annotations
    to_annotate = [m for m in masks if m['annotation'] is None]
    to_annotate = sorted(to_annotate, key=lambda mask: np.max(mask['mean_logit']))
    # this is the only difference with selecting masks randomly!
    to_annotate = to_annotate[:num_annotations]
    num_masks_per_gt_class = np.zeros(num_classes, dtype=int)
    for m in to_annotate:
        m['annotation'] = m['label']
        num_masks_per_gt_class[m['label']] += 1
    print('{} annotated masks'.format(len(to_annotate)))
    print('Per class :')
    for c in range(num_classes):
        print(CLASSES[c], num_masks_per_gt_class[c], 'masks gt',
              np.round(100 * num_masks_per_gt_class[c] / len(to_annotate), decimals=2), '% of masks')
    # no need to return a result, the list of annotated masks, because we may want all the
    # masks to annotate pseudolabels. the list of all masks contains now the annotated masks plus
    # those not annotated


def select_masks_balanced_by_number(masks, num_annotations, num_classes, sort_by_maxlogit):
    assert len(masks) >= num_annotations
    masks_per_predicted_class = {}
    for c in range(num_classes):
        masks_per_predicted_class[c] = []
    for m in masks:
        pred = np.argmax(m['counts_pred'])
        masks_per_predicted_class[pred].append(m)
    if sort_by_maxlogit:
        for c in range(num_classes):
            masks_per_predicted_class[c] = \
                sorted(masks_per_predicted_class[c], key=lambda mask: np.max(mask['mean_logit']))
    else:
        masks_per_predicted_class[c] = np.random.permutation(masks_per_predicted_class[c])
    # seleccionar
    i = 0
    num_masks_per_predicted_class = np.zeros(num_classes, dtype=int)
    num_masks_per_gt_class = np.zeros(num_classes, dtype=int)
    num_pixels_per_gt_class = np.zeros(num_classes, dtype=int)
    num_total_pixels = 0
    num_annotated_masks = 0
    while num_annotated_masks < num_annotations:
        for c in range(num_classes):
            if num_annotated_masks < num_annotations:
                if len(masks_per_predicted_class[c]) > i:
                    mask = masks_per_predicted_class[c][i]
                    mask['annotation'] = mask['label']
                    num_annotated_masks += 1
                    num_masks_per_predicted_class[c] += 1
                    num_masks_per_gt_class[mask['label']] += 1
                    num_pixels_per_gt_class[mask['label']] += mask['area']
                    num_total_pixels += mask['area']
            else:
                break
        i += 1

    print('{} annotated masks'.format(num_annotated_masks))
    print('Per class :')
    for c in range(num_classes):
        print(CLASSES[c], ' : ',
              num_masks_per_predicted_class[c],
              'masks predicted',
              np.round(100 * num_masks_per_predicted_class[c] / num_annotated_masks, decimals=2), '%',
              num_masks_per_gt_class[c],
              ', masks gt',
              np.round(100 * num_masks_per_gt_class[c] / num_annotated_masks, decimals=2), '%',
              ', pixels gt',
              np.round(100 * num_pixels_per_gt_class[c] / num_total_pixels, decimals=2), '%')


def select_masks_balanced_by_pixels(masks, num_annotations, num_classes, sort_by_maxlogit):
    masks_per_predicted_class = {}
    for c in range(num_classes):
        masks_per_predicted_class[c] = []
    for m in masks:
        pred = np.argmax(m['counts_pred'])
        masks_per_predicted_class[pred].append(m)
    if sort_by_maxlogit:
        for c in range(num_classes):
            masks_per_predicted_class[c] = sorted(masks_per_predicted_class[c],
                                                  key=lambda mask: np.max(mask['mean_logit']))
    else:
        print('\n*** not randomized ***\n')
        # for c in range(num_classes):
        #     masks_per_predicted_class[c] = np.random.permutation(masks_per_predicted_class[c])
        #     # just in case they were sorted by image: yes, they are! by radomizing, 500 -> 1200 images 1st round

    num_masks_per_predicted_class = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        num_masks_per_predicted_class[c] = len(masks_per_predicted_class[c])
    annotated_pixels_per_class = np.zeros(num_classes, dtype=int)
    annotated_masks_per_class = np.zeros(num_classes, dtype=int)
    total_masks = 0
    total_pixels = 0
    num_annotated_masks = 0
    while num_annotated_masks < num_annotations:
        # find the class with fewer pixels annotated

        sorted_classes = np.argsort(annotated_pixels_per_class)
        # from classes with fewer to more pixels annotated, find the class with some mask left to annotate
        for c in sorted_classes:
            if annotated_masks_per_class[c] < num_masks_per_predicted_class[c]:
                mask = masks_per_predicted_class[c][annotated_masks_per_class[c]]
                mask['annotation'] = mask['label']
                num_annotated_masks += 1
                annotated_masks_per_class[c] += 1
                annotated_pixels_per_class[c] += mask['area']
                total_masks += 1
                total_pixels += mask['area']
                break

    for c in range(num_classes):
        print(CLASSES[c], ':',
              annotated_masks_per_class[c], 'masks',
              np.round(100 * annotated_masks_per_class[c] / total_masks, decimals=2), '%',
              annotated_pixels_per_class[c], 'pixels',
              np.round(100 * annotated_pixels_per_class[c] / total_pixels, decimals=2), '%')


def annotate_nodes(nodes_to_annotate, masks, max_num_annotations,
                   num_leaves_to_annotate_per_cluster, min_consensus):
    num_annotations = 0
    num_pseudolabels = 0
    num_annotated_nodes = 0
    for node in nodes_to_annotate:
        some_leaf_annotated = False
        pseudolabel, consensus, ids_leaves_to_annotate, gts_leaves = \
            compute_pseudolabel(node, num_leaves_to_annotate_per_cluster)
        # annotate with groundtruth
        for id_leaf, gt_leaf in zip(ids_leaves_to_annotate, gts_leaves):
            mask = masks[id_leaf]
            if mask['annotation'] is None:
                mask['annotation'] = gt_leaf
                num_annotations += 1
                if mask['pseudolabel'] is not None:
                    mask['pseudolabel'] = None
                    num_pseudolabels -= 1

        if consensus >= min_consensus:
            for id_mask in node.ids_leaves:
                # annotate with pseudolabel
                if id_mask not in ids_leaves_to_annotate:
                    mask = masks[id_mask]
                    if mask['annotation'] is None and mask['pseudolabel'] is None:
                        mask['pseudolabel'] = pseudolabel
                        num_pseudolabels += 1
                        some_leaf_annotated = True

        if some_leaf_annotated:
            num_annotated_nodes += 1
        if num_annotations >= max_num_annotations:
            break

    print('total leaves annotated with gt', num_annotations)
    print('nodes to annotate', len(nodes_to_annotate), 'nodes with some leaf annotated with a pseudolabel',
          num_annotated_nodes)
    print('total leaves annotated with pseudolabel', num_pseudolabels)


def select_nodes_balanced_by_pixels(nodes_to_annotate, masks, num_annotations, num_leaves_to_annotate_per_cluster,
                                    min_consensus, num_classes, sort_by_maxlogit):
    assert len(masks) >= num_annotations
    nodes_per_predicted_class = {}
    for c in range(num_classes):
        nodes_per_predicted_class[c] = []
    for node in nodes_to_annotate:
        nodes_per_predicted_class[node.pred].append(node)
    if sort_by_maxlogit:
        for c in range(num_classes):
            nodes_per_predicted_class[c] = sorted(nodes_per_predicted_class[c], key=lambda node: node.max_logit)

    annotated_pixels_per_class = np.zeros(num_classes, dtype=int)
    annotated_nodes_per_class = np.zeros(num_classes, dtype=int)
    annotated_masks_per_class = np.zeros(num_classes, dtype=int)
    total_annotations = 0
    total_pixels = 0
    total_masks = 0
    total_nodes = 0

    while total_annotations < num_annotations:
        # select one node of the predicted class with fewer pixels annotated and with some node left to annotate
        sorted_classes = np.argsort(annotated_pixels_per_class)
        # print('annotated_pixels_per_class', annotated_pixels_per_class)
        # print('sorted classes', sorted_classes)
        node = None
        for c in sorted_classes:
            if annotated_nodes_per_class[c] < len(nodes_per_predicted_class[c]):
                node = nodes_per_predicted_class[c][annotated_nodes_per_class[c]]
                annotated_nodes_per_class[c] += 1
                total_nodes += 1
                break
            else:
                pass
                # print('class ', c, 'has no nodes left to annotate',
                #      annotated_nodes_per_class[c], '>=', len(nodes_per_predicted_class[c]))

            # print('c', c, 'sorted_classes[c]', sorted_classes[c], 'node.pred', node.pred)
        assert node is not None

        annotation, consensus, ids_leaves_to_annotate, gts_leaves = \
            compute_pseudolabel(node, num_leaves_to_annotate_per_cluster)
        # print('annotation', annotation)
        total_annotations += len(ids_leaves_to_annotate)
        if consensus >= min_consensus:
            # do annotate
            for id_mask in node.ids_leaves:
                mask = masks[id_mask]
                mask['annotation'] = annotation

            annotated_masks_per_class[annotation] += len(node.ids_leaves)
            annotated_pixels_per_class[annotation] += node.area
            total_pixels += node.area
            total_masks += len(node.ids_leaves)
        else:
            # even if there's not enough consensus, do not discard the annotated leaves
            for id_leaf, gt_leaf in zip(ids_leaves_to_annotate, gts_leaves):
                mask = masks[id_leaf]
                mask['annotation'] = gt_leaf
                annotated_masks_per_class[gt_leaf] += 1
                annotated_pixels_per_class[gt_leaf] += mask['area']
                total_pixels += mask['area']
                total_masks += 1

    for c in range(num_classes):
        print(CLASSES[c], ':',
              annotated_nodes_per_class[c], 'nodes',
              np.round(100 * annotated_nodes_per_class[c] / total_nodes, decimals=2), '%',
              annotated_masks_per_class[c], 'masks',
              np.round(100 * annotated_masks_per_class[c] / total_masks, decimals=2), '%',
              annotated_pixels_per_class[c], 'pixels',
              np.round(100 * annotated_pixels_per_class[c] / total_pixels, decimals=2), '%')


def add_pseudolabels_threshold(masks, num_classes):
    print('adding pseudos...')
    not_yet_annotated = {}
    for c in range(num_classes):
        not_yet_annotated[c] = []

    print('computing maxlogit of masks not yet annotated')
    for mask in tqdm(masks):
        if mask['annotation'] is None and mask['pseudolabel'] is None:
            # assert mask['pseudolabel'] is None
            # no perque al clustering tambe faig pseudolabels
            pred = np.argmax(mask['counts_pred'])
            not_yet_annotated[pred].append(mask)
            mask['max_logit'] = np.max(mask['mean_logit'])

    # annotate with the predicted class (pseudolabel, then) the most reliable masks in each *predicted* class, separately
    num_bins_max_logits = 50  # 50, 100
    percent_max_logits = 90  # 90, 95
    min_masks_not_yet_annotated = 10  # 10
    print('\n*** {} bins, {} percent, min masks {} ***\n'
          .format(num_bins_max_logits, percent_max_logits, min_masks_not_yet_annotated))
    num_pseudos_per_class = np.zeros(num_classes, dtype=int)
    num_pixels_per_class = np.zeros(num_classes, dtype=int)
    print('annotating pseudos')
    for c in tqdm(range(num_classes)):
        if len(not_yet_annotated[c]) > min_masks_not_yet_annotated:
            max_logits = [mask['max_logit'] for mask in not_yet_annotated[c]]
            hist_max_logits, bins_max_logits = np.histogram(max_logits, bins=num_bins_max_logits)
            thr_max_logit = (bins_max_logits[:-1] \
                [100 * np.cumsum(hist_max_logits) / np.sum(hist_max_logits) > percent_max_logits])[0]
            # abans era 1 i no 0 i el comentari deia " 0 ?"
            for mask in not_yet_annotated[c]:
                if mask['max_logit'] >= thr_max_logit:
                    pseudolabel = np.argmax(mask['counts_pred'])
                    mask['pseudolabel'] = pseudolabel
                    num_pseudos_per_class[c] += 1
                    num_pixels_per_class[c] += mask['area']
        else:
            print('no mask of class {} not yet annotated'.format(c))

    num_pseudolabeled_masks = 0
    for c in range(num_classes):
        num_pseudolabeled_masks += num_pseudos_per_class[c]
        print(CLASSES[c], ':', num_pseudos_per_class[c], 'pseudolabel annotations',
              np.round(100 * num_pseudos_per_class[c] / sum(num_pseudos_per_class), decimals=2), '%',
              num_pixels_per_class[c], 'pixels',
              np.round(100 * num_pixels_per_class[c] / sum(num_pixels_per_class), decimals=2), '%')
    print('total number of pseudolabeled masks', num_pseudolabeled_masks)
    print('done')


def add_pseudolabels_lca(masks, max_lca_dist):
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

    root, nodes = cluster_features(masks)
    annotated_leaves = [leaf for leaf in nodes if leaf.is_leaf if leaf.annotation is not None]
    # pero no miro si ja te un pseudolabel, perque suposo que vinc d'anotar per maxlogit i no n'hi ha

    total_pseudos = 0
    for i, leaf in enumerate(annotated_leaves):
        common_ancestor = leaf
        # print('i',i, 'leaf', leaf.id)
        for j in range(1, max_lca_dist):
            # print('j',j)
            if not common_ancestor.is_root:
                common_ancestor = common_ancestor.parent
                # print('j', j, 'common_ancestor', common_ancestor.id)
                descendants = LevelOrderIter(common_ancestor, maxlevel=max_lca_dist - j + 1)
                for desc in descendants:
                    if desc.is_leaf \
                            and desc.annotation is None \
                            and desc.pseudolabel is None:
                        #desc.annotation = leaf.annotation  # per no reanotar mes tard
                        desc.pseudolabel = leaf.annotation # per distingir si cal entre anotacions gt i pseudos=propagades
                        # CAL ANOTAR LA MASCARA CORRESPONENT, SI NO NO FEM RES!!
                        desc.mask['pseudolabel'] = leaf.annotation
                        # print('added leaf', desc.id)
                        total_pseudos += 1
                        num_masks_per_class[desc.pseudolabel] += 1
                        num_pixels_per_class[desc.pseudolabel] += desc.area
                        if desc.pseudolabel != desc.gt:
                            num_mask_errors_per_class[desc.gt] += 1
                            num_pixel_errors_per_class[desc.gt] += desc.area

    print('total_pseudos', total_pseudos)
    print('num_masks_per_class', num_masks_per_class)
    print('num_mask_errors_per_class', num_mask_errors_per_class)
    print('% error masks')
    for c in range(num_classes):
        if num_masks_per_class[c] > 0:
            err_mask = np.round(100 * num_mask_errors_per_class[c] / num_masks_per_class[c], decimals=2)
            print('class', c, err_mask, '%')
        else:
            print('class', c, 'has no annotated masks')
    print('num_pixels_per_class', num_pixels_per_class)
    print('num_pixel_errors_per_class', num_pixel_errors_per_class)
    print('% error pixels')
    for c in range(num_classes):
        if num_pixels_per_class[c] > 0:
            err_pix = np.round(100 * num_pixel_errors_per_class[c] / num_pixels_per_class[c], decimals=2)
            print('class', c, err_pix, '%')
        else:
            print('class', c, 'has no annotated pixels')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-processed-masks', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--num-annotations', type=int, required=True)
    parser.add_argument('--path-out', required=True)

    parser.add_argument('--select-random-masks', action="store_true", required=False)
    parser.set_defaults(select_random_masks=False)

    parser.add_argument('--select-masks-sorted-by-maxlogit', action="store_true", required=False)
    parser.set_defaults(select_masks_sorted_by_maxlogit=False)

    parser.add_argument('--add-pseudolabels-threshold', action="store_true", required=False)
    parser.set_defaults(add_pseudolabels_threshold=False)
    parser.add_argument('--add-pseudolabels-lca', action="store_true", required=False)
    parser.set_defaults(add_pseudolabels_lca=False)
    parser.add_argument('--max-lca-dist', type=int, required=False, default=2)

    parser.add_argument('--select-masks-balanced-by-number', action="store_true", required=False)
    parser.set_defaults(select_masks_balanced_by_number=False)
    parser.add_argument('--select-masks-balanced-by-pixels', action="store_true", required=False)
    parser.set_defaults(select_masks_balanced_by_pixels=False)
    parser.add_argument('--sort-by-maxlogit', action="store_true", required=False)
    parser.set_defaults(sort_by_maxlogit=False)

    parser.add_argument('--select-nodes-sorted', action="store_true", required=False)
    parser.add_argument('--select-nodes-balanced-by-pixels', action="store_true", required=False)
    parser.add_argument('--percent-dists', type=float, required=False)
    parser.add_argument('--num-leaves-to-annotate-per-cluster', type=int, required=False)
    parser.add_argument('--min-consensus', type=float, required=False)
    parser.set_defaults(select_nodes=False)

    args = parser.parse_args()
    assert args.select_random_masks \
           + args.select_masks_sorted_by_maxlogit \
           + args.select_masks_balanced_by_number \
           + args.select_masks_balanced_by_pixels \
           + args.select_nodes_sorted \
           + args.select_nodes_balanced_by_pixels == 1

    if args.select_nodes:
        assert args.percent_dists is not None
        assert args.min_consensus is not None
        assert args.num_leaves_to_annotate_per_cluster is not None

    return args


if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.path_out):
        print('file {} already exists, exiting script annotate_masks.py'.format(args.path_out))
        exit()

    masks = load_masks(args.path_processed_masks)
    # remove pseudolabel annotations from previous round, if any (may be now it's the first round)
    for m in masks:
        m['pseudolabel'] = None
    print('cleared pseudolabels from previous round, if any')

    if args.select_random_masks:
        select_random_masks(masks, args.num_annotations)
    elif args.select_masks_sorted_by_maxlogit:
        select_masks_sorted_by_maxlogit(masks, args.num_annotations, args.num_classes)
    elif args.select_masks_balanced_by_number:
        select_masks_balanced_by_number(masks, args.num_annotations, args.num_classes,
                                        args.sort_by_maxlogit)
    elif args.select_masks_balanced_by_pixels:
        select_masks_balanced_by_pixels(masks, args.num_annotations, args.num_classes,
                                        args.sort_by_maxlogit)
    elif args.select_nodes_sorted or args.select_nodes_balanced_by_pixels:
        root, nodes = cluster_features(masks)

        if False:
            # sort by DECREASING distance between the two children
            min_num_leaves = 16
            selected_parents = [node for node in nodes if not node.is_leaf and len(node.ids_leaves) >= min_num_leaves]
            nodes_to_annotate = sorted(selected_parents, key=lambda p: -p.distance)
        else:
            # sort by INCREASING maxlogit
            min_num_leaves = 32  # 2
            selected_parents = [node for node in nodes if not node.is_leaf and len(node.ids_leaves) >= min_num_leaves]
            nodes_to_annotate = sorted(selected_parents, key=lambda node: node.max_logit)

        if args.select_nodes_sorted:
            annotate_nodes(nodes_to_annotate, masks, args.num_annotations,
                           args.num_leaves_to_annotate_per_cluster,
                           args.min_consensus)
        elif args.select_nodes_balanced_by_pixels:
            select_nodes_balanced_by_pixels(nodes_to_annotate, masks, args.num_annotations,
                                            args.num_leaves_to_annotate_per_cluster,
                                            args.min_consensus, args.num_classes,
                                            args.sort_by_maxlogit)

    if args.add_pseudolabels_threshold:
        add_pseudolabels_threshold(masks, args.num_classes)

    if args.add_pseudolabels_lca and (args.max_lca_dist > 1) :
        add_pseudolabels_lca(masks, args.max_lca_dist)

    num_annotated_masks = 0
    num_pseudolabeled_masks = 0
    for m in masks:
        if m['annotation'] is not None:
            num_annotated_masks += 1
        if m['pseudolabel'] is not None:
            num_pseudolabeled_masks += 1
        assert (m['annotation'] is not None) + (m['pseudolabel'] is not None) <= 1
    print('{} annotated masks in this plus previous rounds'.format(num_annotated_masks))
    print('{} pseudolabeled masks in this round'.format(num_pseudolabeled_masks))


    save_annotated_masks(masks, args.path_out)

"""
# of each note the class with highest score, and second and third highest scores
pred1 = np.array([np.argsort(node.counts_pred_pixels)[::-1][0] for node in nodes_to_annotate])
pred2 = np.array([np.argsort(node.counts_pred_pixels)[::-1][1] for node in nodes_to_annotate])
pred3 = np.array([np.argsort(node.counts_pred_pixels)[::-1][2] for node in nodes_to_annotate])
# not exactly gt because only pixels have gt, and masks almost since they are quite homogenous,
# and a parent node has 2 or more descendant masks
gt = np.array([np.argmax(node.counts_gt_pixels) for node in nodes_to_annotate])
print(np.sum(gt == pred1) / len(gt),
    (np.sum(gt == pred1) + np.sum(gt == pred2)) / len(gt),
    (np.sum(gt == pred1) + np.sum(gt == pred2) + np.sum(gt == pred3)) / len(gt))
nodes_per_class = {}
for c in range(args.num_classes):
  nodes_per_class[c] = []

for i,node in enumerate(nodes_to_annotate):
  nodes_per_class[pred1[i]].append(node)
  nodes_per_class[pred2[i]].append(node)
  nodes_per_class[pred3[i]].append(node)

nodes_to_annotate2 = []
for c in range(args.num_classes):
  nodes_per_class[c] = sorted(nodes_per_class[c], key=lambda node: node.max_logit)
  #print('pre-selected nodes', CLASSES[c], len(nodes_per_class[c]))
  nodes_to_annotate2.extend(nodes_per_class[c][:100])
# shuffle
nodes_to_annotate2 = np.random.permutation(nodes_to_annotate2)
"""

"""
# clustering with fastcluster
import fastcluster

t0 = time()
fn = lambda u, v: np.dot(u,v)
hac = fastcluster.linkage_vector(features.copy(), method='centroid', metric='euclidean') # method='single', metric=fn)
# centroid = linkage average de AgglomerativeClustering ? tots dos poden ser single i ward
t1 = time()
print('hac fastcluster : {} features, {} secs.'.format(len(features), t1-t0))
# hac fastcluster : 21766 features, 166.5477912425995 secs. mes que amb sklearn
"""

"""
def display_results(nodes, nodes_to_annotate, num_classes, plot_figures):
    leaves = [node for node in nodes if node.is_leaf]
    parents = [node for node in nodes if not node.is_leaf]
    # error degut a SAM = al fet d'agafar tots els pixels d'una regio SAM com de la mateixa classe i agafar com
    # a classe groundtruth el label majoritari gt d'entre aquests pixels = quin percentatge de pixels, entre totes les
    # classes, tenen un label gt diferent del label agafat per la regio
    sam_percent_error = 100 * np.array([(leaf.counts_gt.sum() - leaf.counts_gt[leaf.label]) / leaf.counts_gt.sum()
                                        for leaf in leaves])
    for p in [1, 5, 10, 15, 20, 25,]:
        print('% regions SAM amb < {}% pixels de classe no majoritaria : {}%'
              .format(p, np.round(100*(sam_percent_error <= p).sum() / len(leaves), decimals=1)))

    # quin es l'error si anotem cada node amb la classe majoritaria de les seves mascares=fulles ?
    # error a nivell pixel, una part del qual ja es el d'haver assignat un label a tota la mascara
    parents_to_annotate = [node for node in nodes_to_annotate if not node.is_leaf]
    leaves_to_annotate = [node for node in nodes_to_annotate if node.is_leaf]
    print('percent_max_logits, percent_dists', args.percent_max_logits, args.percent_dists)
    print('nombre nodes a anotar', len(nodes_to_annotate))
    print('nombre nodes pare', len(parents_to_annotate))
    print('nombre nodes fulla', len(leaves_to_annotate))
    print('nombre de masks totals anotades = fulles descendents : ',
          np.sum([node.num_leaves for node in nodes_to_annotate]))
    print('nombre de pixels totals anotats : {} M pixels'.
          format(np.sum([node.area for node in nodes_to_annotate]) // 1e6))
    print('nombre de nodes per classe anotats',
          np.bincount([node.label for node in nodes_to_annotate], minlength=num_classes))

    pixels_per_class = {}
    regions_per_class = {}
    for c in range(num_classes):
        pixels_per_class[c] = 0
        regions_per_class[c] = 0
        for node in nodes_to_annotate:
            if node.label == c:
                pixels_per_class[c] += node.counts_gt[c]
                regions_per_class[c] += node.num_leaves

    print('nombre de regions i pixels / classe')
    for c in range(num_classes):
        print(c, '\t', regions_per_class[c], '\t', pixels_per_class[c], '\t', CLASSES[c])

    # node.label == arg max node.counts_gt
    percent_errors = [100 * (node.counts_gt.sum() - node.counts_gt[node.label]) / node.counts_gt.sum() for node in
                      nodes_to_annotate]
    for p in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        print('% nodes a anotar amb error < {}% pixels : {}%'.format(p, np.round(
            100 * np.sum(np.array(percent_errors) < p) / len(nodes_to_annotate), decimals=1)))

    if plot_figures:
        #TODO: canviar per histograma acumulat
        plt.figure()
        plt.hist(sam_percent_error, bins=50)
        plt.title('% error degut a SAM')
        plt.show(block=False)

        plt.figure()
        plt.hist([np.max(l.max_logit) for l in leaves], bins=50, histtype='step', label='leaves')
        plt.hist([np.max(p.max_logit) for p in parents], bins=50, histtype='step', label='parents')
        plt.hist([np.max(n.max_logit) for n in nodes], bins=50, histtype='step', label='nodes')
        #max_logits = np.array([leaf.max_logit for leaf in leaves])
        #x = np.linspace(max_logits.min(), max_logits.max(), 100)
        #y = norm.pdf(x, loc=np.mean(max_logits), scale=np.std(max_logits))
        #y2 = np.histogram(max_logits, bins=50)[0].max() * y / y.max()
        #plt.plot(x, y2)
        plt.legend()
        plt.title('histograma max_logit')
        plt.show(block=False)

        plt.figure()
        plt.hist(hac.distances_, bins=100)
        #x = np.linspace(hac.distances_.min(), hac.distances_.max(), 100)
        #y = norm.pdf(x, loc=np.mean(hac.distances_), scale=np.std(hac.distances_))
        #y2 = np.histogram(hac.distances_, bins=100)[0].max() * y / y.max()
        #plt.plot(x, y2)
        plt.title('distancies entre els 2 fills d''un pare')
        plt.show(block=False)

        plt.figure()
        plt.hist(percent_errors, bins=50)
        plt.title('histograma % nodes a anotar amb error < % pixels')
        plt.show(block=False)
"""
