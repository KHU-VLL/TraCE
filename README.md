<div align="center">

# TraCE: Trajectory-Aware Concepts for Explainable Video Understanding

Wooil Lee<sup>1†</sup>, Junpyo Hong<sup>1†</sup>, Jongseo Lee<sup>1</sup>, Tuomas Oikarinen<sup>2</sup>, Tsui-Wei Weng<sup>2</sup>, Jinwoo Choi<sup>1*</sup>

**<sup>1</sup>Kyung Hee University, <sup>2</sup>University of California San Diego**

<sup>†</sup>Equal contribution, <sup>*</sup>Corresponding author

**NeurIPS 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white)](https://leewooil.github.io/TraCE_project_page/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://leewooil.github.io/TraCE_project_page/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<img src="assets/teaser.png" alt="TraCE teaser" width="100%">

</div>

PyTorch implementation of **TraCE**, a concept-based framework that turns optical-flow trajectories into explanatory concepts. Existing video explanations say *that* motion matters but not *which* motion. TraCE clusters trajectories into **trajectory concepts** and pairs them with **object** and **scene** concepts, so one typed concept interface explains both **action recognition** (sparse linear classifier) and **video question answering** (sparse concept matching).

## Installation

```bash
conda create -n trace python=3.10 -y
conda activate trace

git clone https://github.com/KHU-VLL/TraCE.git && cd TraCE
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Preparation

| Dataset | Task | Annotations |
|:--|:--|:--|
| [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) (Mutual) | AR, VQA | `annotations/ntu_mutual` |
| [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) (Chiral) | AR, VQA | `annotations/chiral_ssv2` |
| [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) | AR, VQA | `annotations/ucf101` |
| [KTH](https://www.csc.kth.se/cvap/actions/) | AR | `annotations/kth` |
| [TOMATO](https://github.com/yale-nlp/TOMATO) | VQA | `annotations/tomato` |
| [STAR](https://github.com/csbobby/STAR) | VQA | `tools/data/build_star_qa.py` |

Set `VIDEO_ROOT` to each dataset's video folder; video names in the annotation files are relative to it.

Place the backbone weights in `pretrained/`:

```
pretrained/
├── videomae_ssv2_finetune.pth          # VideoMAE ViT-B/16 (SSv2)
├── videomae_ucf101_finetune.pth        # VideoMAE ViT-B/16 (UCF-101)
├── InternVideo2-stage2_1b-224p-f4.pt   # InternVideo2 Stage-2 1B
└── bert-large-uncased/
```

## Training

Each `scripts/<dataset>.sh` runs concept discovery, action recognition and VQA. Pass a stage name (`trajectory`, `text`, `ar`, `vqa`) to run one stage.

```bash
VIDEO_ROOT=data/ntu_mutual/videos bash scripts/ntu_mutual.sh        # all stages
VIDEO_ROOT=data/ntu_mutual/videos bash scripts/ntu_mutual.sh ar     # action recognition only
```

Per-dataset settings are in `configs/`. Trajectory discovery uses [WAFT](https://github.com/princeton-vl/WAFT) and [U²-Net](https://github.com/xuebinqin/U-2-Net) (`WAFT_ROOT`, `U2NET_ROOT`).

## Evaluation & Explanation

```bash
# accuracy and Top-K / Random-K trajectory-concept removal
python tools/eval_ar.py --bundle runs/ar/kth --features work/kth/features_ar/val_videomae.pt \
    --annotation annotations/kth/val.csv --removal

# top concepts, intervention and trajectory overlay for one video
python tools/explain_ar.py --bundle runs/ar/chiral_ssv2 --video clip.mp4 \
    --remove_top_trajectory 3 --overlay 3 --out_dir outputs/explain_ar
```

`tools/eval_vqa.py` and `tools/explain_vqa.py` work the same way for VQA. Demos are in [`notebooks/`](notebooks).

Checkpoints and cached features will be released soon.

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgements

This project builds upon [DANCE](https://github.com/KHU-VLL/DANCE), [Label-free CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM), [MM-CBM](https://github.com/Trustworthy-ML-Lab/Multi-Modal-CBM), [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [InternVideo2](https://github.com/OpenGVLab/InternVideo).

## Citation

```bibtex
@inproceedings{lee2026trace,
  title     = {TraCE: Trajectory-Aware Concepts for Explainable Video Understanding},
  author    = {Lee, Wooil and Hong, Junpyo and Lee, Jongseo and Oikarinen, Tuomas and Weng, Tsui-Wei and Choi, Jinwoo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
