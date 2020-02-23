# deepneedle
This project was done under EG3301R in NUS as part of my engineering undergraduate requirement.
Paper: https://www.springerprofessional.de/en/ultrasound-needle-segmentation-and-trajectory-prediction-using-e/17588942

Abstract
Purpose
Ultrasound (US)-guided percutaneous kidney biopsy is a challenge for interventionists as US artefacts prevent accurate viewing of the biopsy needle tip. Automatic needle tracking and trajectory prediction can increase operator confidence in performing biopsies, reduce procedure time, minimize the risk of inadvertent biopsy bleedings, and enable future image-guided robotic procedures.
Methods
In this paper, we propose a tracking-by-segmentation model with spatial and channel “Squeeze and Excitation” (scSE) for US needle detection and trajectory prediction. We adopt a light deep learning architecture (e.g. LinkNet) as our segmentation baseline network and integrate the scSE module to learn spatial information for better prediction. The proposed model is trained with the US images of anonymized kidney biopsy clips from 8 patients. The contour is obtained using the border-following algorithm and area calculated using Green formula. Trajectory prediction is made by extrapolating from the smallest bounding box that can capture the contour.
Results
We train and test our model on a total of 996 images extracted from 102 short videos at a rate of 3 frames per second from each video. A set of 794 images is used for training and 202 images for testing. Our model has achieved IOU of 41.01%, dice accuracy of 56.65%, F1-score of 36.61%, and root-mean-square angle error of 13.3∘. We are thus able to predict and extrapolate the trajectory of the biopsy needle with decent accuracy for interventionists to better perform biopsies.
Conclusion
Our novel model combining LinkNet and scSE shows a promising result for kidney biopsy application, which implies potential to other similar ultrasound-guided biopsies that require needle tracking and trajectory prediction.
