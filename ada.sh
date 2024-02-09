# conda activate openmmlab
# bash ./work_dirs/ada/ada.sh

SAMPLE_IMAGES=4  # 4 10 2 5 100
SAMPLE_MASKS=1   #  1  2 1 4  40
SEED=1
NUM_ROUNDS=3 # 3, 5
NUM_ANNOTATIONS=64000 # budget
NUM_CLASSES=19
#PERCENT_DISTS=80
#NUM_LEAVES_TO_ANNOTATE_PER_CLUSTER=16
#MIN_CONSENSUS=1.0
MAX_LCA_DIST=(2 2 4) # 2->1303 pseudos, la meitat de classe 0: 157, 2: 244, 13: 265 = road, building, car
MAX_ITERS=10000

GT_SHAPE='1024 2048'
IGNORE_LABEL=255
PATH_ALL_MASKS='../sam/all_masks_cityscapes_train_0.86_0.92_400_1000.pkl'
PATH_SAMPLED_MASKS='./work_dirs/ada/sampled_masks_'$SAMPLE_IMAGES'_'$SAMPLE_MASKS'.pkl'
DIR_MASKS='../sam/masks_0.86_0.92_400/cityscapes/train'
DIR_IMAGES='data/cityscapes/leftImg8bit/train'
DATA_ROOT='data/cityscapes'
SUBDIR_ANNOTATIONS='annotations'
ROOT_DIR_ANNOTATED_IMAGES=$DATA_ROOT'/'$SUBDIR_ANNOTATIONS
PATH_TEMPLATE_CONFIG_TRAINING='./work_dirs/ada/segformer_prototypes_mit-b5_1xb1.py'


python ./work_dirs/ada/sample_masks.py \
  --sample-images $SAMPLE_IMAGES \
  --sample-masks $SAMPLE_MASKS \
  --seed $SEED \
  --path-all-masks $PATH_ALL_MASKS \
  --path-out $PATH_SAMPLED_MASKS


for ((ROUND = 2; ROUND <= $NUM_ROUNDS; ROUND++)); do

  ANNOTATION_ID=$SAMPLE_IMAGES'_'$SAMPLE_MASKS'_'$ROUND
  PATH_PROCESSED_MASKS='./work_dirs/ada/processed_masks_'$ANNOTATION_ID'.pkl'
  PATH_ANNOTATED_MASKS='./work_dirs/ada/annotated_masks_'$ANNOTATION_ID'.pkl'
  DIR_ANNOTATED_IMAGES=$ROOT_DIR_ANNOTATED_IMAGES'/'$ANNOTATION_ID
  PATH_ANN_FILE=$DIR_ANNOTATED_IMAGES'.txt'
  PATH_CONFIG_TRAINING='./work_dirs/ada/segformer_prototypes_mit-b5_1xb1_'$ROUND'.py'

  if [[ $ROUND == 1 ]]; then
    PATH_CONFIG_SAMPLING='work_dirs/segformer_prototypes/segformer_prototypes_mit-b5_1xb1-20k_16065.py'
    PATH_CHECKPOINT='work_dirs/segformer_prototypes_mit-b5_2x1_1024x1024_160k_gta/iter_160000.pth'
    #PSEUDOLABELS='--add-pseudolabels-lca --max-lca-dist 2' # ''
  else
    PATH_SAMPLED_MASKS='./work_dirs/ada/annotated_masks_'$SAMPLE_IMAGES'_'$SAMPLE_MASKS'_'$(($ROUND-1))'.pkl'
    PATH_CONFIG_SAMPLING='work_dirs/ada/segformer_prototypes_mit-b5_1xb1_'$(($ROUND-1))'.py'
    PATH_CHECKPOINT=$(cat 'work_dirs/ada/round_'$(($ROUND-1))'/last_checkpoint')
    #PSEUDOLABELS='--add-pseudolabels-lca --max-lca-dist '$MAX_LCA_DIST #--add-pseudolabels-thr'
  fi

  CUDA_VISIBLE_DEVICES=0 python ./work_dirs/ada/process_sampled_masks.py \
  --path-sampled-masks $PATH_SAMPLED_MASKS \
  --path-checkpoint $PATH_CHECKPOINT \
  --path-config $PATH_CONFIG_SAMPLING \
  --dir-masks $DIR_MASKS \
  --dir-images $DIR_IMAGES \
  --num-classes $NUM_CLASSES \
  --path-out $PATH_PROCESSED_MASKS

  python ./work_dirs/ada/annotate_masks.py \
    --path-processed-masks $PATH_PROCESSED_MASKS \
    --num-classes $NUM_CLASSES \
    --num-annotations $(($NUM_ANNOTATIONS/$NUM_ROUNDS)) \
    --path-out $PATH_ANNOTATED_MASKS \
    --select-masks-sorted-by-maxlogit \
    --add-pseudolabels-lca --max-lca-dist ${MAX_LCA_DIST[ROUND - 1]}
    #
    # add one of the following:
    #--select-random-masks
    #
    #--select-masks-sorted-by-maxlogit
    #
    #--select-masks-balanced-by-number
    #--sort-by-maxlogit # default false
    #
    #--select-masks-balanced-by-pixels
    #--sort-by-maxlogit # default false
    #
    #--select-nodes-sorted-by-maxlogit \
    #--percent-dists ${PERCENT_DISTS[ROUND - 1]} \
    #--min-consensus $MIN_CONSENSUS \
    #--num-leaves-to-annotate-per-cluster $NUM_LEAVES_TO_ANNOTATE_PER_CLUSTER

  python ./work_dirs/ada/annotate_images.py \
      --path-annotated-masks $PATH_ANNOTATED_MASKS \
      --dir-images $DIR_IMAGES \
      --dir-masks $DIR_MASKS \
      --gt-shape $GT_SHAPE \
      --ignore-label $IGNORE_LABEL \
      --dir-out $DIR_ANNOTATED_IMAGES \
      --dir-out-images $DIR_ANNOTATED_IMAGES/images \
      --dir-out-annotations $DIR_ANNOTATED_IMAGES/annotations \
      --path-ann-file $PATH_ANN_FILE

  python ./work_dirs/ada/make_config_training.py \
    --work-dir './work_dirs/ada/round_'$ROUND \
    --path-checkpoint $PATH_CHECKPOINT \
    --ann-file $SUBDIR_ANNOTATIONS'/'$ANNOTATION_ID'.txt' \
    --max-iters $MAX_ITERS \
    --path-template-config $PATH_TEMPLATE_CONFIG_TRAINING \
    --path-config-out $PATH_CONFIG_TRAINING \
    --img-path $SUBDIR_ANNOTATIONS'/'$ANNOTATION_ID'/images' \
    --seg-map-path $SUBDIR_ANNOTATIONS'/'$ANNOTATION_ID'/annotations'
    #--img-path 'leftImg8bit/train' \
    #--seg-map-path $SUBDIR_ANNOTATIONS'/'$ANNOTATION_ID'

  echo 'Start training of round' $ROUND '...'
  CUDA_VISIBLE_DEVICES=0,1 PORT=29500 source tools/dist_train.sh $PATH_CONFIG_TRAINING 2 >/dev/null 2>&1
  # discard output, the interesting output (mIoU) goes to the log file. 2>&1 means error
  # messages go to the console
  echo 'done'

done

: '
python -i tools/analysis_tools/analyze_logs.py \
  work_dirs/ada/results/20231129_175238_1round_30k_masks_sorted_by_maxlogit_64k_masks/vis_data/vis_data/20231129_175238.json \
  work_dirs/ada/results/20231126_222359_1round_15k_masks_sorted_by_maxlogit_64k_masks/vis_data/20231126_222359.json \
  work_dirs/ada/results/20231127_110740_1round_15k_masks_balanced_by_pixels_64k_masks/20231127_110740/vis_data/20231127_110740.json \
  work_dirs/ada/results/20231126_095312_1round_5k_random_masks_64k_masks/vis_data/20231126_095312.json \
  work_dirs/ada/results/20231130_125853_1round_30k_masks_sorted_by_maxlogit_64k_masks_pseudolabels/vis_data/20231130_125853.json \
  work_dirs/ada/results/pseudos_rounds/fusio_3_rounds.json
'
