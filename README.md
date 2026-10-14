# LLT

Implementation of **Memory Matters: Boosting Training-Free Zero-Shot Temporal Action Localization with a Learnable Lookup Table**, CVPR 2026.

[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.pdf)

LLT accumulates action-positive visual candidates across test videos, adapts lookup items with a lightweight residual, and updates text prototypes through test-time residual learning and momentum. The CoCa backbone stays frozen. No supervised training stage is required.

## Setup

Use Linux, Bash, Python 3.10, and a CUDA-capable NVIDIA GPU.

```bash
conda create -n llt python=3.10 -y
conda activate llt
cd LLT
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Run

Run THUMOS14 split 0 in the 50% seen / 50% unseen or 75% seen / 25% unseen setting:

```bash
bash auto_50.sh
bash auto_75.sh
```

Evaluate all ten class partitions:

```bash
LAST_SPLIT=9 bash auto_50.sh
LAST_SPLIT=9 bash auto_75.sh
```

ActivityNet uses the corresponding model and data configuration:

```bash
DATASET=anet bash auto_75.sh
DATASET=anet LAST_SPLIT=9 bash auto_50.sh
```

Pass Hydra overrides through either script, or call the Python entry directly:

```bash
CUDA_VISIBLE_DEVICES=0 bash auto_75.sh data.feature_path=/data/my_thumos_features
python src/run.py data=anet model=anet model.setting=75 model.split=0 paths.anet_fps=/data/activitynet_fps.json
```


## Citation

```bibtex
@InProceedings{Jiang_2026_CVPR,
  author = {Jiang, Han and Tang, Haoyu and Mu, Xiaoxuan and Li, Chen and Zhu, Jihua},
  title = {Memory Matters: Boosting Training-Free Zero-Shot Temporal Action Localization with a Learnable Lookup Table},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026}
}
```
