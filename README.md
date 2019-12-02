
# [MultitaskNet](https://github.com/imvinod/MultitaskNet) implementation in PyTorch
<br>

![MultitaskNet](/home/vinod/Git_Workspace/MultitaskNet/images/multitasknet.png  "MultitaskNet")


## Prerequisites
- Linux
- Python 3.7.0
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.4.1.post2 and dependencies from http://pytorch.org
- Clone this repo:
```bash
git clone https://github.com/imvinod/MultitaskNet
cd MultitaskNet
pip install -r requirements.txt
```
## Dataset preparation
### sunrgbd dataset
- Download and untar the [preprocessed sunrgbd](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/sun/sunrgbd.tar.gz) dataset under ```/datasets/sunrgbd```

## FuseNet train/test

### visdom visualization
- To view training errors and loss plots, set `--display_id 1`, run `python -m visdom.server` and click the URL http://localhost:8097
- Checkpoints are saved under `./checkpoints/sunrgbd/`

### train & test on sunrgbd
```bash
python train.py --dataroot datasets/sunrgbd --dataset sunrgbd --name sunrgbd

python test.py --dataroot datasets/sunrgbd --dataset sunrgbd --name sunrgbd --epoch 100
```

## Results
* We use the training scheme defined in MultitaskNet
* Loss is weighted for SUNRGBD dataset
* Learning rate is set to 0.0001 for SUNRGBD dataset
* Results can be improved with a hyper-parameter search

## More details coming up  !