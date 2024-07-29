# Motion-aware 3D Gaussian Splatting

> [Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction](https://arxiv.org/abs/2403.11447)
>
> Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, Houqiang Li

## Installation

```shell
git clone https://github.com/jasongzy/MAGS
cd MAGS
git submodule update --init --recursive

conda create -n mags python=3.10
conda activate mags
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
```

## Data Preparation

1. Follow the instruction of [4DGaussians](https://github.com/hustvl/4DGaussians#data-preparation) to download and preprare the datasets.

2. Use [Unimatch](https://github.com/autonomousvision/unimatch) or [RAFT](https://github.com/princeton-vl/RAFT) to obtain the optical flow between frames.

```
├── data
│   |── hypernerf
│       ├── interp
│           ├── chickchicken
│             ├── camera
│             ├── flow
│       ├── misc
│           ├── ...
│       ├── virg
│           ├── ...
│   |── dynerf
│       ├── cook_spinach
│         ├── cam00
│           ├── images
│             ├── 0000.png
│             ├── 0001.png
│             ├── 0002.png
│             ├── ...
│           ├── flow
│             ├── 0000_pred.npy
│             ├── 0001_pred.npy
│             ├── 0001_pred_bwd.png
│             ├── ...
│         ├── cam01
│           ├── ...
│       ├── cut_roasted_beef
|       ├── ...
```

## Training

E.g., to train the model on scene HyperNeRF-`chickchicken`:

```shell
python train.py -s data/hypernerf/interp/chickchicken --configs arguments/hypernerf/default.py --expname mags/hypernerf/interp-chicken
```

## Citation

```bibtex
@article{guo2024motion,
  title={Motion-aware {3D} Gaussian Splatting for Efficient Dynamic Scene Reconstruction},
  author={Guo, Zhiyang and Zhou, Wengang and Li, Li and Wang, Min and Li, Houqiang},
  journal={arXiv preprint arXiv:2403.11447},
  year={2024}
}
```
