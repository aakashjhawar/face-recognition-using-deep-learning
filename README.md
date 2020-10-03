# Face Recognition using OpenCV
 - Create dataset of face images
 - - Detect faces using ```deploy.prototxt``` and ```res10_300x300_ssd_iter_140000.caffemodel```. (Learn more about [face detection](https://github.com/aakashjhawar/face-detection))
 - Extract face embeddings for each face present in the image using pretrained [OpenFace](https://cmusatyalab.github.io/openface/) model ```openface_nn4.small2.v1.t7```. 
 - Train a SVM model on the face embeddings to recognize faces 

## Overview of OpenFace for a single input image
1. Detect faces with a pre-trained models from dlib or OpenCV.
2. Transform the face for the neural network. This repository uses dlib's real-time pose estimation with OpenCV's affine transformation to try to make the eyes and bottom lip appear in the same location on each image.
3. Use a deep neural network to represent (or embed) the face on a 128-dimensional unit hypersphere. The embedding is a generic representation for anybody's face. Unlike other face representations, this embedding has the nice property that a larger distance between two face embeddings means that the faces are likely not of the same person. This property makes clustering, similarity detection, and classification tasks easier than other face recognition techniques where the Euclidean distance between features is not meaningful.
4. Apply clustering or classification techniques to the features to complete the face recognition task.

*Read more about [OpenFace](https://cmusatyalab.github.io/openface/)*
![Working of OpenCV Face detector](https://github.com/aakashjhawar/face-recognition-using-opencv/blob/master/images/openface.jpg)

## Getting Started
How to use
```    
git clone https://github.com/aakashjhawar/face-recognition-using-opencv
cd face-recognition-using-opencv
```
 - Create dataset of face images.
 - Place the face images in dataset folder.
 - Extract facial embeddings.
```python extract_embeddings.py```
 - Train the SVM model
```python train_model.py```
 - Test the model
```python recognize_video.py```

## Prerequisites

- Python 3.5
- OpenCV
```
sudo apt-get install python-opencv
```

## Results 

#### Detect and recognize faces from video camera-
![Result](https://github.com/aakashjhawar/face-recognition-using-opencv/blob/master/images/output1.png)


