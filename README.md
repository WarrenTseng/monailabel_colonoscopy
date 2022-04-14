# An Example: Implement 2D colonoscopy segmentation model into MONAI Label

![monailabel_capture](https://user-images.githubusercontent.com/24987707/162153621-2386c5c2-7333-4f28-897e-e10e56680a73.JPG)

## Environment
The app was developed with the official MONAI docker container (https://hub.docker.com/r/projectmonai/monai, v0.8.1) and MONAI Label v0.4.dev2213. </br>
MONAI Label installation:
```bash
pip install monailabel-weekly==v0.4.dev2213
```

## Pre-trained model
Please put the pre-trained models of PraNet-19 and Res2Net weights into the folder colonoscopy_app/model/. </br>
The pre-trained models can be downloaded from the PraNet repo: https://github.com/DengPingFan/PraNet#31-trainingtesting

## Sample data
From Kvasir-SEG dataset https://datasets.simula.no/kvasir-seg/

Note:
- Replace monailabel transforms.py (located in /your/pythonlib/path/of/monailabel/deepedit/multilabel/transforms.py) with the provided transforms.py
- Replace monai dice.py (located in /your/pythonlib/path/of/monai/losses/dice.py with the provided dice.py
</br>

The full repo with pre-trained models can be downloaded here: https://drive.google.com/file/d/1tetR6TtWcsp_g6mkHmsk8kFvDuDpiJdA/view?usp=sharing

## Step to Build Monai on Clara AGX Dev Kit
The pre-build docker image for PyTorch/TorchVision/Monai/MonaiLabel is provided on docker hub.
```bash
docker pull eahung/cagx_monai
```
SW version in the image
- Ubuntu 20.04
- CUDA 11.2.0
- CUDNN 8
- PyTorch 1.8.0
- TorchVision 0.9
- TorchAudio 0.8
- Monai 0.8.1

#### Steps to build the image (It takes several hours and a large amount of disk space to build the image)

1. Pull docker image nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

2. Build PyTorch and TorchVision
    -  Suggested reed: https://qengineering.eu/install-pytorch-on-jetson-nano.html

3. Build Monai (https://docs.monai.io/en/stable/installation.html)
```bash
BUILD_MONAI=1 
pip install --no-build-isolation git+https://github.com/Project-MONAI/MONAI#egg=monai
```
4. Build Monai Label (https://docs.monai.io/en/stable/installation.html)
```bash
pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel
```

#### Contributors
Eddie Huang, NVIDIA, tzungchih@nvidia.com
Eason Hung, NVIDIA, eahung@nvidia.com






