## Low-resolution Face Detection Code

<br/>

### Dependencies
* [Python 3.6](https://www.continuum.io/downloads)
* [PyTorch 0.4.1](http://pytorch.org/)

<br/>

### Usage

### 1. Download widerface dataset
(1) Create widerface directory
```bash
$ mkdir widerface
$ cd widerface
```

(2) Download data in the widerface directory 
* Training images: https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing \
* Validation images: https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing <br/>

<br/>

### 2. Install packages
```bash
$ cd face_detection
$ pip install -r requirements.txt
```

<br/>

### 3. Evaluate pretrained model

```bash
$ python evaluation.py --checkpoint "weights/resnet101_checkpoint_20.pth" \
                       --base_model "resnet101" \
                       --prob_thresh 0.8 \
                       --dataset-root PATH_OF_WIDER_FACE_DATASET/widerface 
```

<br/>

### 3. Train your own model
```bash
$ python main.py --epochs 50 \
                 --dataset-root PATH_OF_WIDERFACE_DATASET/widerface 
```
<br/>

## Reference
Hu, Peiyun, and Deva Ramanan. "Finding tiny faces." Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017.

<br/>
