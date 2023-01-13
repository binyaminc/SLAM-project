# Overview
This project operates a SLAM (Simultaneous localization and mapping) system. It uses images from a set of rectified stereo cameras with their clibration matrices, and produces a map of the percieved road.

The project consists of 3 components:
1. Detection and matching of the keypoints in a pair of images, and creation of 3D points using triangulation.
2. Finding the pose of each pair of cameras using Ransac algorithm with PnP as inner model
3. Refining the results using bundle adjustment
