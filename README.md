# Overview
This project operates a SLAM (Simultaneous localization and mapping) system. It uses images from a set of rectified stereo cameras with their clibration matrices, and produces a map of the percieved road.

The project consists of 3 components:
1. Detection and matching of the keypoints in a pair of images, and creation of 3D points using triangulation.
2. Finding the pose of each pair of cameras using Ransac algorithm with PnP as inner model
3. Refining the results using bundle adjustment

# Requirements
In order to use this program, this are the requirements:
* Linux operating system
* Pyhon 3.8
* Opencv-python
* Gtsam
* Database with rectified images, and calibration matrices. ground truth poses are required for evaluating the results. I use a dataset of KITTI, can be found [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The database tree is this:


###

    .
    ├───code
    │   ├───...                     # The files of the project - part1, part2...
    └───data
        └───dataset
            ├───poses               # ground truth poses of the cameras. files like 00.txt 
            └───sequences
                ├───00              # specific dataset
                    ├───image_0     # the images from the left camera
                    ├───image_1     # the images from the right camera
                    └───calib.txt   # clibration matrices of the cameras
