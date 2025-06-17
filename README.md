# 6D-pose-estimation


This project presents a full pipeline for estimating the 6D pose of objects using RGB-D images. It starts with object detection using YOLOv8 and then refines the object’s position and orientation with a custom deep learning model called Enhanced RCVPose. Everything—from preparing the data and training the models to evaluating results and visualizing the poses—is clearly documented in the 6D-Pose-Estimation.ipynb notebook.

---

## 📌 Description

The goal of this project is to accurately determine the **3D rotation and translation (6D pose)** of objects in an image relative to the camera. This is achieved through a two-stage process:

1. **Object Detection**  
   A pre-trained YOLOv8s model detects and localizes objects in the RGB images, producing bounding boxes.

2. **Pose Estimation**  
   The Enhanced RCVPose model uses RGB-D data to predict the 6D pose. It leverages:
   - A ResNet50 backbone
   - Feature Pyramid Network (FPN)
   - Attention modules  
   for robust multi-scale feature extraction.

The project is tailored for the **LineMOD dataset** and includes full scripts for data preprocessing, training, and evaluation.

---


## ✨ Features

-  **YOLOv8s Integration** for object detection
-  **EnhancedRCVPose Model** with ResNet50, FPN, and Attention
-  **Data Preprocessing**: radius map generation, pose file creation, and more
-  **Training & Evaluation**: full loops with detailed metrics

---

## 📁 Dataset

This project uses the **LineMOD** dataset — a standard benchmark for 6D pose estimation.

Preprocessing includes:
-  Unzipping preprocessed LineMOD data
-  Organizing data into object-specific folders
-  Generating `Outside9.npy` keypoints via Farthest-Point Sampling
-  Creating `poseXXXXXX.npy` files from `gt.yml`
-  Formatting RGB, mask, and depth files
-  Normalizing the data

The dataset is split into:
- 70% training
- 20% validation
- 10% testing

---

## 🛠️ Installation

The notebook is built for use in **Google Colab**. Dependencies are installed within the notebook or You can install all the dependencies at once, you can run the following command in your terminal:

```bash
pip install -r requirements.txt
```
---
## Model Architecture

###  Object Detection
- **YOLOv8s**: A lightweight and accurate object detector from Ultralytics.
- Used to extract bounding boxes around objects from RGB images.

###  Pose Estimation – EnhancedRCVPose
- A deep model built on:
  -  **ResNet50** backbone
  -  **Feature Pyramid Network (FPN)** for multi-scale features
  -  **Attention Modules** to focus on relevant object areas
- **Inputs**:
  - RGB-D crops (image patches within YOLO bounding boxes)
  - 9 radius maps from `Outside9.npy`
- **Outputs**:
  - 3D translation vector (x, y, z)
  - 4D rotation quaternion (x, y, z, w)
  - 9 predicted radius maps (supervised via MSE)
---

##  Results

###  Object Detection (YOLOv8)
Evaluated using:
-  Precision
-  Recall
-  mAP (mean average precision)

###  Pose Estimation (EnhancedRCVPose)
Evaluated using:
-  **Translation RMSE**: Mean error in meters for object location
-  **Rotation Error**: Angle difference (degrees) between predicted and GT rotation
-  **Points MSE**: Per-pixel error on radius maps
-  **ADD**: Average Distance between transformed 3D model points

>  Best model selected using validation loss  

---

### 📸 Sample Outputs

Below is an example of YOLO predictions and 6D pose estimation results using the EnhancedRCVPose model on the validation sets:

![Yolo Prediction](/sample_output/val_batch2_pred.jpg)
![Pose Estimation Prediction](/sample_output/pose_estimate_pred.png)
---

