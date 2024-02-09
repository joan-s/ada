from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv


checkpoint = 'work_dirs/segformer_prototypes/b5_80325/iter_20000.pth'
config = 'work_dirs/segformer_prototypes/b5_80325/segformer_prototypes_mit-b5_1xb1-20k_80325.py'
image = 'data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'

# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device='cuda')
model.cfg.model.test_cfg.mode='whole'
# desa el config com un atribut mes del model

# test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
img = mmcv.imread(image)
result = inference_model(model, img)
feat = model.decode_head.normalized_features([1024,2048])

# result.seg_logits.data.shape is torch.Size([20, 1024, 2048])
# result.pred_sem_seg.data.shape is torch.Size([1, 1024, 2048])

out_file = None
show_result_pyplot(
    model,
    img,
    result,
    title=image.split('/')[-1],
    opacity=0.5,
    draw_gt=False,
    show=False if out_file is not None else True,
    out_file=out_file)

import numpy as np
masks = np.load('../sam/masks_0.86_0.92_400/cityscapes/train/aachen/aachen_000115_000019_masks.npz', allow_pickle=True).get('masks')
masks[0].keys() # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])

model.decode_head.conv_seg # Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
# per obtenir les features 256 assignar identity a conv_seg i fer inference_model() o encode_decode() ?
# encode_decode() vol inputs: Tensor, batch_img_metas: List[dict]
# pels logits ja esta perque result ja els te
# model.extract_feat nomes aplica el backbone


