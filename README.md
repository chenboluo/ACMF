# Attention Consistency Refined Masked Frequency Forgery Representation for Generalizing Face Forgery Detection


## Dependencies

* torch 1.10.0+cu113
* timm  0.4.12

More details about dependencies are shown in requirements.txt.

## Dataset Preparation

* We include the dataset loaders for WildDeepfake and Celeb-DF. You can enter the dataset website to download the original data. https://github.com/deepfakeinthewild/deepfake-in-the-wild and https://github.com/yuezunli/celeb-deepfakeforensics
* For WildDeepfake, you should first extract the facial images from the sequences and store them. We use RetinaFace to do this.

## Usage

### Training & testing

To test the model in the Celeb-DF in first stage, run the following script in your console. The model will start training and return the AUC at each epoch.
```
python test-cele.py 
```

First-stage-Celeb-DF-resutl.txt is the result.

To test the model in the WDF in first stage, run the following script in your console. The model will start training and return the AUC at each epoch.
```
python test-wdf.py 
```
First-stage-WDF-resutl.txt is the result.

To test the model in the WDF in second stage, run the following script in your console. The model will start training and return the AUC at each epoch.
```
python Train-second-stage.py
```
second-stage-wdf.txt is the result.

### Pre-trained models
You can load the pre-trained model in test files.
You can download the pre-trained model from here.
https://pan.baidu.com/s/1sULCnA-5WVtGHtfCjc9_PQ 
key：ozcc
