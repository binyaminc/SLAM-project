import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import part1

DATA_PATH = r'D:\SLAM\exercises\VAN_ex\data\dataset05\sequences\05\\'

FILTER_RATIO = 0.75
NUM_MATCHES_SHOW = 200
DISTANCE_THREASHOLD = 400  # previously, 100


def get_keypoints_and_matches(img1, img2, rectified=True):

    # --------- keypoints ---------
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # --------- BFMatcher with default params ---------
    bf = cv2.BFMatcher()
    double_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    matches = []
    for m, n in double_matches:
        if m.distance < DISTANCE_THREASHOLD and m.distance < FILTER_RATIO * n.distance:
            matches.append(m)  # not appending m as a list, as opposed to what I did in part1.py

    # --------- filter matches if height differs in more than 2 pixels ---------
    if rectified:
        matches = list(
            filter(lambda m: abs(keypoints_1[m.queryIdx].pt[1] - keypoints_2[m.trainIdx].pt[1]) <= 2, matches))

    return keypoints_1, keypoints_2, matches


def proc(img1, img2, rectified=True):

    # --------- keypoints ---------
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # --------- BFMatcher with default params ---------
    bf = cv2.BFMatcher()
    double_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    matches = []
    for m, n in double_matches:
        if m.distance < DISTANCE_THREASHOLD and m.distance < FILTER_RATIO * n.distance:
            matches.append([m])  # append m as a list, so that we get list of lists as required in drawMatchesKnn

    # --------- filter matches if height differs in more than 2 pixels ---------
    if rectified:
        # img1 = queryImage, img2 = trainImage
        matches = list(
            filter(lambda m: abs(keypoints_1[m[0].queryIdx].pt[1] - keypoints_2[m[0].trainIdx].pt[1]) <= 2, matches))


    # --------- triangulation ---------
    matched_kp1, matched_kp2 = part1.get_matched_kps_from_matches(matches, keypoints_1, keypoints_2)

    k, m1, m2 = part1.read_cameras()

    # matters of reshaping, hopefully
    np_kp1 = np.array([kp.pt for kp in matched_kp1])
    np_kp2 = np.array([kp.pt for kp in matched_kp2])

    # OpenCV implementation
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, np_kp1.T, np_kp2.T)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]

    part1.plot_3d_points(points_3d_hom)

    return keypoints_1, keypoints_2, matched_kp1, matched_kp2, points_3d_hom


def get_cloud(matched_kps_l, matched_kps_r, k, m1, m2):
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, matched_kps_l.T, matched_kps_r.T)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]

    return points_3d_hom


def main():
    img1, img2 = part1.read_images(0)
    kps1, kps2, matched_kp1, matched_kp2, points3d = proc(img1, img2)

    img1, img2 = part1.read_images(1)
    kps1, kps2, matched_kp1, matched_kp2, points3d = proc(img1, img2)


if __name__ == "__main__":
    main()