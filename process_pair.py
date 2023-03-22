import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = r'../data/dataset/sequences/05//'

FILTER_RATIO = 0.75
NUM_MATCHES_SHOW = 200
DISTANCE_THREASHOLD = 400  # previously, 100
HEIGHT_THRESH = 2


def get_keypoints_and_matches(img1, img2, rectified=True):
    # find keypoints
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = get_keypoints(img1, img2) 

    # apply BFMatcher with default params
    matches = get_matches(descriptors_1, descriptors_2)
    
    # filter matches by height (takes only matches that differ in less than 2 pixels)
    if rectified:
        matches = filter_matches_with_height(matches, keypoints_1, keypoints_2)
        
    return keypoints_1, keypoints_2, matches


def get_keypoints(img1, img2):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    return keypoints_1, descriptors_1, keypoints_2, descriptors_2


def get_matches(descriptors_1, descriptors_2):
    bf = cv2.BFMatcher()
    double_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    matches = []
    for m, n in double_matches:
        if m.distance < DISTANCE_THREASHOLD and m.distance < FILTER_RATIO * n.distance:
            matches.append(m)  # not appending m as a list, as opposed to what I did in part1.py

    return matches


def filter_matches_with_height(matches, keypoints_1, keypoints_2):
    """
    filter the matches according to the geight of the matched keypoints. 
    Note: img1 = queryImage, img2 = trainImage
    """
    filtered_matches = list(
        filter(lambda m: abs(keypoints_1[m.queryIdx].pt[1] - keypoints_2[m.trainIdx].pt[1]) <= HEIGHT_THRESH, matches))
    
    return filtered_matches


def get_cloud(matched_kps_l, matched_kps_r, k, m1, m2):
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, matched_kps_l.T, matched_kps_r.T)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]

    return points_3d_hom


def main():
    # img1, img2 = part1.read_images(0)
    # kps1, kps2, matched_kp1, matched_kp2, points3d = proc(img1, img2)

    # img1, img2 = part1.read_images(1)
    # kps1, kps2, matched_kp1, matched_kp2, points3d = proc(img1, img2)
    pass


if __name__ == "__main__":
    main()


"""
def proc(img1, img2, rectified=True):

    keypoints_1, keypoints_2, matches = get_keypoints_and_matches(img1, img2, rectified)

    # --------- triangulation ---------
    matched_kp1, matched_kp2 = part1.get_matched_kps(matches, keypoints_1, keypoints_2)

    k, m1, m2 = part1.read_cameras()

    # get the 2d_locations of kps, in numpy array
    np_kp1 = np.array([kp.pt for kp in matched_kp1])
    np_kp2 = np.array([kp.pt for kp in matched_kp2])

    # OpenCV implementation
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, np_kp1.T, np_kp2.T)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]

    part1.plot_3d_points(points_3d_hom)

    return keypoints_1, keypoints_2, matched_kp1, matched_kp2, points_3d_hom


def my_triangulatePoints(P, Q, kp1, kp2):

    points_4d_hom = np.empty(shape=(4,0))

    # looping over all the points
    for i in range(len(kp1)):

        # looping over the cameras and keypoint
        A = np.empty(shape=(0, 4))

        P_equation_vector_1 = P[2] * kp1[i][0] - P[0]
        P_equation_vector_2 = P[2] * kp1[i][1] - P[1]
        A = np.vstack([A, P_equation_vector_1, P_equation_vector_2])

        Q_equation_vector_1 = Q[2] * kp2[i][0] - Q[0]
        Q_equation_vector_2 = Q[2] * kp2[i][1] - Q[1]
        A = np.vstack([A, Q_equation_vector_1, Q_equation_vector_2])

        # SVD on A, return the last column in V
        u,s,v = np.linalg.svd(A)

        points_4d_hom = np.c_[points_4d_hom, v[-1]]

    return points_4d_hom
"""