# Graph-Based Motion Prediction for Abnormal Action Detection

This is an official pytorch implementation of [Graph-Based Motion Prediction for Abnormal Action Detection](https://dl.acm.org/doi/abs/10.1145/3444685.3446316). In this work,  we propose human motion prediction for abnormal action detection. We employ sequence of human poses to represent human motion, and detect irregular behavior by comparing the predicted pose with the actual pose detected in the frame.

## Introduction

This project provides the codes of MP-GCN (motion prediction GCN), which predicts future human poses in the next frame. The framework of MG-GCN is illustrated below:

<img src="MP-GCN.png" style="zoom: 80%;float:left" />

## Main Results

### Detection results on NJUST-Anomaly

|       Methods        |  AUC  |
| :------------------: | :---: |
|   FramePrediction    | 0.560 |
|    Autoregression    | 0.601 |
| SkeletonTrajectories | 0.709 |
|    Memory-guided     | 0.563 |
| Our method (MP-GCN)  | 0.732 |


## Environment

The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA 2080Ti GPU.

## Quick start

### Installation

#### MP-GCN

1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/). **Note that if you use pytorch's version < v1.0.0, you should following the instruction at https://github.com/Microsoft/human-pose-estimation.pytorch to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${MPGCN_ROOT}.
3. Install dependencies:

```python
#Python3 Pytorch
#You can create a new conda environment for convenience.

pip install -r requirements.txt

cd torchlight; python setup.py install; 
```

### Data Preparation

Our NJUST-Anomaly dataset contains 137 videos, including 107 training video clips and 30 testing video clips. All of the video clips can be downloaded [here](https://pan.baidu.com/s/1TT8Qn0Q8nkhxOeY5_QaiRg). Extraction code: wpbj. The directory tree should look like this:

   ```
   ${NJUST-Anomaly data}
   |-- train
       `--|--videos
          | |-- 1.mp4
          | |-- 2.mp4
          | |-- 3.mp4
          | |-- ...
   `-- test
       `--|-- videos
          | |-- 1.mp4
          | |-- 2.mp4
          | |-- 3.mp4
          | |-- ...    
   ```

Several data processing steps are needed to prepare the data for training and testing MP-GCN. After doing all these steps, the directory tree should be like this:

```
   ${MPGCN_ROOT}
   ├── pose
   ├── jsondata
   ├── label
   ├── mp_gcn
   ├── pose_choose
   └── pose_xy
   ```

The details are listed bellow:

1. Detect and track human poses in each video clip:

   We use [CenterNet](https://github.com/xingyizhou/CenterNet)+[Deep_sort](https://github.com/nwojke/deep_sort)+[HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation) to get the human poses. Other human detector, multi-object tracking method, and human pose estimation method can be used.
   For conveniece, we provide the detected human poses, please download from [here](). Put these pose data in ${MPGCN_ROOT}/pose. Or you detect and track the human poses yourself by following the steps in the paper.

2. Screen poses according to the pose scores

   Choose the most confident poses according to the scores provided by the human pose estimation algorithm. Put the choosed poses in ${MPGCN_ROOT}/pose_choose. 

   ```
   python choose_pose.py
   ```
   
   We also provide the choosed poses, please download from [here]().

3. Motion decomposition

   Do motion decomposition for the choosed poses. Put the processed data in ${MPGCN_ROOT}/pose_xy. 

   ```
   python pose_xy.py
   ```
   
   Or please download our processed data from [here]().

4. Generate data for MP-GCN

   Generate the data for MP-GCN. This step will generate two files, the first is jsondata.json, please put it in ${MPGCN_ROOT}/jsondata, the second is label.json, please put in ${MPGCN_ROOT}/label.

   ```
   python json2json_oks.py
   ```
   
   Or please download our generated MP-GCN input data [here]().

5. Prepare data for **train** or **test**.

   Use the data from step 4 to generate data for train or test. Put the generated data in ${MPGCN_ROOT}/mp_gcn. 

   ```python
   python tools/kinetics_gendata.py
   # line 82 & line 83
   # change the path according to the generated data from step 4.
   ```
   
   Or please download our generated training or testing data [here]().
   

## Training and Testing 

##### Train on NJUST-Anomaly.

```python
python main.py recognition -c config/st_gcn/<dataset>/train.yaml

# change path in train.yaml according to your data
```

The trained model will be put in ${MPGCN_ROOT}/mp_gcn/work_dir.

##### Testing on NJUST-Anomaly. Our trained model is provided in ${MPGCN_ROOT}/models.

```python
python main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml

# change path in test.yaml according to your data 
```

The test result will be saved in test_result.pkl, which can be found in ${MPGCN_ROOT}/mp_gcn/work_dir.

## Calculate the AUC value

1. Prepare the data for AUC evaluation. Please download the mask file from [here]().

   Put the test result (test_result.pkl) in the directory ${MPGCN_ROOT}/mp_gcn/. For the NJUST-Anomaly test set, the result of MP-GCN can be downloaded [here](). Put the file label.json (the same file as in ${MPGCN_ROOT}/label) and the mask file in the directory ${MPGCN_ROOT}/mp_gcn/. The directory tree should be like this.

   ```
   ${MPGCN_ROOT}
   |-- mp_gcn
      ├── mask
      |   |-- 1.mpy
      |   `-- 2.mpy
      |   `-- ...
      ├── label.json
      └── test_result.pkl
   ```

2. calculate AUC value

   ```python
   python oks.py
   ```

### Ciation

If you use our code or models in your research, please cite with:

```
@inproceedings{10.1145/3444685.3446316,
author = {Tang, Yao and Zhao, Lin and Yao, Zhaoliang and Gong, Chen and Yang, Jian},
title = {Graph-Based Motion Prediction for Abnormal Action Detection},
year = {2021},
isbn = {9781450383080},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3444685.3446316},
doi = {10.1145/3444685.3446316},
booktitle = {Proceedings of the 2nd ACM International Conference on Multimedia in Asia},
articleno = {63},
numpages = {7},
keywords = {graph convolutional network, abnormal action detection, motion prediction},
location = {Virtual Event, Singapore},
series = {MMAsia '20}
}
```

### Acknowledgement

The codes are developed based on the opensource of [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md) .
