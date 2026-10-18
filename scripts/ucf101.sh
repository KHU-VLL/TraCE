#!/bin/bash
# TraCE pipeline for UCF-101. Run the stages you need; each stage reads the
# outputs of the previous one. Edit the paths below.
set -e
cd "$(dirname "$0")/.."
DATASET=ucf101
VIDEO_ROOT=${VIDEO_ROOT:-data/$DATASET/videos}         # raw videos (see README)
WORK=${WORK:-work/$DATASET}                            # intermediate files
PRETRAINED=${PRETRAINED:-pretrained}                   # backbones (see README)
WAFT_ROOT=${WAFT_ROOT:-external/WAFT}; U2NET_ROOT=${U2NET_ROOT:-external/U-2-Net}
QWEN=${QWEN:-Qwen/Qwen3-VL-8B-Instruct}
IV2="--iv2_ckpt $PRETRAINED/InternVideo2-stage2_1b-224p-f4.pt --bert_path $PRETRAINED/bert-large-uncased"
STAGE=${1:-all}
run() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# 1. Trajectory concepts: optical flow -> trajectories -> foreground masks -> FINCH -> soft labels
if run trajectory; then
  CFG=configs/discovery/trajectory_$DATASET.yaml
  python tools/discovery/extract_flow_waft.py --config $CFG --waft_root $WAFT_ROOT --ckpt $PRETRAINED/tar-c-t.pth \
      --dav2_ckpt $PRETRAINED/depth_anything_v2_vits.pth --video_root $VIDEO_ROOT --video_list annotations/$DATASET/train.csv annotations/$DATASET/val.csv --output_dir $WORK/flow
  python tools/discovery/build_trajectories.py --config $CFG --input_dir $WORK/flow --output_dir $WORK/stacks
  python tools/discovery/foreground_masks_u2net.py --config $CFG --u2net_root $U2NET_ROOT --weights $PRETRAINED/u2net.pth \
      --video_root $VIDEO_ROOT --video_list annotations/$DATASET/train.csv annotations/$DATASET/val.csv --output_dir $WORK/masks
  python tools/discovery/cluster_trajectories.py --config $CFG --input_dir $WORK/stacks --mask_root $WORK/masks \
      --flow_root $WORK/flow --sample_filter_csv annotations/$DATASET/train.csv --output_root $WORK/clusters
  CLUSTER_DIR=$(ls -d $WORK/clusters/global_partition_*/ | tail -1)  # latest run
  python tools/discovery/trajectory_soft_labels.py --config $CFG --cluster_dir $CLUSTER_DIR --train_csv annotations/$DATASET/train.csv \
      --val_csv annotations/$DATASET/val.csv --val_stack_dir $WORK/stacks --mask_root $WORK/masks --flow_root $WORK/flow --out_dir $WORK/trajectory_labels
  python tools/discovery/export_trajectory_concepts.py --cluster_dir $CLUSTER_DIR --dataset $DATASET --out_dir $WORK/trajectory_concepts
fi

# 2. Object / scene concepts -> concepts/$DATASET/{object,scene}_candidates.txt (NTU RGB+D Mutual is included)
if run text; then
  for P in object_v1 scene_v1; do
    python tools/discovery/generate_text_candidates_qwen3vl.py --config configs/discovery/text_$DATASET.yaml --prompt $P \
        --model_path $QWEN --video_root $VIDEO_ROOT --out_dir $WORK/text
  done
  python tools/discovery/filter_text_concepts.py --results $WORK/text/object_v1_results.jsonl --out_txt concepts/$DATASET/object_candidates.txt
  python tools/discovery/filter_text_concepts.py --results $WORK/text/scene_v1_results.jsonl --out_txt concepts/$DATASET/scene_candidates.txt
fi

# 3. Action recognition: features -> concept layers + sparse classifier -> evaluation
if run ar; then
  for SPLIT in train val; do
    python tools/extract_ar_features.py --annotation annotations/$DATASET/$SPLIT.csv --split $SPLIT --video_root $VIDEO_ROOT \
        --sampling dense --videomae_ckpt $PRETRAINED/videomae_ucf101_finetune.pth $IV2 --out_dir $WORK/features_ar
  done
  python tools/train_ar.py --config configs/ar/$DATASET.yaml --features_dir $WORK/features_ar \
      --trajectory_labels $WORK/trajectory_labels $IV2 --out_dir runs/ar/$DATASET
  python tools/eval_ar.py --bundle runs/ar/$DATASET --features $WORK/features_ar/val_videomae.pt \
      --annotation annotations/$DATASET/val.csv --removal
fi

# 4. Video QA: features -> concept layers + text projector -> evaluation
if run vqa; then
  python tools/extract_vqa_features.py --qa_json annotations/$DATASET/qa_train.json --format ar_qa --video_root $VIDEO_ROOT $IV2 --out_dir $WORK/features_vqa/train
  python tools/extract_vqa_features.py --qa_json annotations/$DATASET/qa_val.json --format ar_qa --video_root $VIDEO_ROOT $IV2 --out_dir $WORK/features_vqa/val
  python tools/train_vqa.py --config configs/vqa/$DATASET.yaml --train_dir $WORK/features_vqa/train --val_dir $WORK/features_vqa/val \
      --trajectory_labels $WORK/trajectory_labels $IV2 --out_dir runs/vqa/$DATASET
  python tools/eval_vqa.py --bundle runs/vqa/$DATASET --split_dir $WORK/features_vqa/val --removal
fi
