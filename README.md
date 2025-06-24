<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Endo-4DGS: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting</h1>
  <div align="center">
    <h4><a href="https://lastbasket.github.io/"><strong>Yiming Huang</strong></a> *, <a href="https://beileicui.github.io/"><strong>Beilei Cui</strong></a> *, <a href="https://longbai-cuhk.github.io/"><strong>Long Bai</strong></a> *, Ziqi Guo, <a href="https://xumengyaamy.github.io/"><strong>Mengya Xu</strong></a>, <a href="https://www.ee.cuhk.edu.hk/en-gb/people/academic-staff/professors/prof-ren-hongliang"><strong>Hongliang Ren</strong></a> </h3>
    <h3> Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024 </h2>
    <h3 align="center"> || <a href="https://arxiv.org/pdf/2401.16416">Paper</a> || <a href="https://arxiv.org/abs/2401.16416">Arxiv</a> || </h3>
    <div align="center"></div>
  </div>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./assets/archi_1.jpg" alt="Logo" width="85%">
  </a>
</p>

## TODO
- [x] EndoNeRF
- [x] StereoMIS
- [x] Pretrained checkpoints


## Environments

We build the Python environment using [Anaconda](https://www.anaconda.com/download/):
1. Install the CUDA toolkit on ubuntu following [this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and then:
```shell
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

2. Install the Python environment
```shell
git clone https://github.com/lastbasket/Endo-4DGS.git
cd Endo-4DGS
git submodule update --init --recursive
conda create -n ED4DGS python=3.8
conda activate ED4DGS

pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics
```

## Datasets
We used two datasets for training and evaluation.

### EndoNeRF

We used EndoNeRF dataset provided by [Yuehao Wang](https://docs.google.com/forms/d/e/1FAIpQLSfM0ukpixJkZzlK1G3QSA7CMCoOJMFFdHm5ltCV1K6GNVb3nQ/viewform).

### StereoMIS
We used StereoMIS dataset from [Michel Hayoz](https://zenodo.org/records/7727692)

The data structure is as follows:

```
./data
├── endonerf
│    ├── cutting_tissues_twice
│    │    └── ...
│    ├── pulling_soft_tissues
│    │    └── ...
│    └── ...
└── stereomis
    └── ...
```

How to prepare stereomis:
```
# 1. unzip StereoMIS_0_0_1.zip to StereoMIS_0_0_1
# 2. run the script
bash prepare_stereomis.sh
```
```
# 3. arrange the files as
./data
└── stereomis
    └── split_1
    │    │    └── images
    │    │    └── poses_bounds.npy
    │    │    └── ...
```
```
# 4. you may also visualize the trajectory using
cd stereomis
python visualize_traj.py
```
### Preparing the Pretrained Depths from Depth-Anything
Download [depth_anything_vits14.onnx](https://github.com/fabio-sim/Depth-Anything-ONNX/releases) and place in:
./submodules/depth_anything/weights/depth_anything_vits14.onnx
```bash
sh prepare_depth.sh
```
## Training


```bash
sh train.sh
```
More configurations can be found in arguments/$DATASET.py

## Evaluation

```bash
# Render the testing result
sh render.sh
# Evaluation
sh eval.sh
```

## Inferring the pretrained Gaussians
Download [pretrained_gaussians](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209042_link_cuhk_edu_hk/ERUYhd6E4INEj2IwxREjZxIBNMU9iMozbnPEIcWse-pBbA?e=z05Cfw) and unzip in the project directory:
```bash
# Render the testing result e.g. endonerf/pulling
python render.py --model_path pretrained_gaussians/endonerf/pulling --pc --skip_video --skip_train --configs arguments/endonerf.py
```

## Cite
```
@inproceedings{huang2024endo,
  title={Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={197--207},
  year={2024},
  organization={Springer}
}

```
## Acknowledgement

Thanks the authors for their works: [StereoMIS](https://arxiv.org/abs/2304.08023v1), [diff-gaussian-rasterization-depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth), [EndoNeRF](https://github.com/med-air/EndoNeRF), [4DGaussians](https://github.com/hustvl/4DGaussians), [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX).
