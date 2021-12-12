Computer Vision Course Project, WPI

## Problem Statement: 
Do anything interesting with a Rubik's cube

## Content:
1. Detection of Rubik's cube 
2. Augumented Reality
3. Pose Estimation

## Detection of Rubik's cube using ORB feature matching and Homography:

<p float="left">
  <img src="media/detect.gif" width="400" />
  <img src="media/detect_kalman.gif" width="400" /> 
</p>
Detection of Rubik's cube using ORB feature matching and Homograpy (left). After Kalman Filter (right)

## Augmented Reality over the Rubik's cube using Perspective Projection:

<p float="left">
  <img src="media/spiderman_ar.gif" width="400" />
  <img src="media/cube_ar.gif" width="400" /> 
</p>
Augumented objects from waveform object files to the face of Rubik's cube

## Pose Estimation of the camera with respect to Rubik's cube using Perspective Projection:

<p float="left">
  <img src="media/pose_estimation.gif" />
</p>
The yaw is very noisy, this is one bug that needs to be fixed. Kalman filter can be used to reduce the estimation noise.

## References
1. [Plane Tracking](https://github.com/opencv/opencv/blob/4.x/samples/python/plane_tracker.py "Plane Tracking")
2. [Augumented Reality](https://github.com/jayantjain100/Augmented-Reality "Augumented Reality")
3. [YCB Dataset](https://www.ycbbenchmarks.com/ "YCB Dataset")

