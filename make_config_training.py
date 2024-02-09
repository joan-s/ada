import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--path-checkpoint', required=True)
    parser.add_argument('--img-path', required=True)
    parser.add_argument('--seg-map-path', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--max-iters', required=True)
    parser.add_argument('--path-template-config', required=True)
    parser.add_argument('--path-config-out', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    lines_to_add = [
        "work_dir = '{}'\n".format(args.work_dir),
        "load_from = '{}'\n".format(args.path_checkpoint),
        "img_path = '{}'\n".format(args.img_path),
        "seg_map_path = '{}'\n".format(args.seg_map_path),
        "ann_file = '{}'\n".format(args.ann_file),
        "max_iters = {}\n".format(args.max_iters),
    ]
    with open(args.path_template_config, 'r') as f:
        lines = f.readlines()

    new_lines = lines_to_add + lines

    with open(args.path_config_out, 'w') as f:
        f.writelines(new_lines)

    print('made config file {}'.format(args.path_config_out))
