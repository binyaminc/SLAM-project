import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
import part1
import process_pair
import random
import cv2
import numpy as np
import itertools
from math import log2, ceil
import pickle

THRESHOLD = 1.5  # 2
PAIRS = 2760
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255,255,0)
ORANGE = (0, 69, 255)

GROUND_TRUTH_LOCATIONS_PATH = r'D:\SLAM\exercises\VAN_ex\data\dataset05\poses\05.txt'


@dataclass
class demi_match:
    def __init__(self, distance, imgIdx, queryIdx, trainIdx):
        self.distance: float = distance
        self.imgIdx: int = imgIdx
        self.queryIdx: int = queryIdx
        self.trainIdx: int = trainIdx


class Track:
    new_id = itertools.count()

    def __init__(self):
        self.TrackId = next(Track.new_id)

        # dictionary to connect pairId in which the track appear, and the match index
        # example - entry can be 3 : 8, which means that in pair 3, the track appears in match 8 (img_l[8] and img_r[8])
        self.PairId_MatchIndex = {}


class Pair:
    def __init__(self, pair_id: int, img_l, img_r):
        self.PairId = pair_id
        self.img_l = img_l
        self.img_r = img_r

        kps_l, kps_r, self.matches = process_pair.get_keypoints_and_matches(img_l, img_r)

        matched_kps_l, matched_kps_r, self.matched_indeces_kp_l, self.matched_indeces_kp_r = part1.get_matched_kps_from_matches_with_matched_indeces([[m] for m in self.matches], kps_l, kps_r)
        # convert to type that can be serializable
        self.matches = [demi_match(m.distance, m.imgIdx, m.queryIdx, m.trainIdx) for m in self.matches]

        # I'm interested only in kps that are matched to the other camera
        self.kps_l = np.array([kp.pt for kp in matched_kps_l])
        self.kps_r = np.array([kp.pt for kp in matched_kps_r])

        # points_3d, in world coordinates
        self.cloud = None  # process_pair.get_cloud(matched_kps_l, matched_kps_r, *part1.read_cameras())

        # TODO: check if to do one variable "transformation", and if to split to global and relative
        self.extrinsic_left = None
        self.extrinsic_right = None
        self.relative_extrinsic_left = None

        # dictionary to connect match indexes to TrackId.
        # (note: match index i - connects kps_l[i] and kps_r[i])
        self.matchIndex_TrackId = {}

    def get_trackIds(self):
        return list(self.matchIndex_TrackId.values())


class Database:
    def __init__(self, file_path=None):

        if not file_path or not os.path.exists(file_path):
            self.Tracks = {}
            self.Pairs = {}
        else:
            self.Pairs, self.Tracks = self.deserialize(file_path)

    def feature_location(self, PairId, TrackId):
        pair = self.pairs[PairId]
        track = self.tracks[TrackId]

        match_index = track.PairId_MatchIndex[PairId]
        return pair.kps_l[match_index], pair.kps_r[match_index]

    def serialize(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.Pairs, self.Tracks), f, pickle.HIGHEST_PROTOCOL)

    def deserialize(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def main():

    """
    database = Database('data - select_better_PnP_or_P3P.pkl')
    pairs = database.Pairs
    tracks = database.Tracks

    show_camera_coords(database)
    """

    database = Database()
    pairs = database.Pairs
    tracks = database.Tracks

    # create pair0
    pair0 = Pair(0, *part1.read_images(0))
    intrinsic, pair0.extrinsic_left, pair0.extrinsic_right = part1.read_cameras()
    pair0.relative_extrinsic_left = pair0.extrinsic_left

    pairs[pair0.PairId] = pair0

    # going over the next cameras. for each 2 pairs - find common kps, find Rt + create Tracks
    for i_pair in range(1, PAIRS+1):
        print(f"------------ pair number {i_pair} ------------")
        prev = pairs[i_pair - 1]
        curr = Pair(i_pair, *part1.read_images(i_pair))

        # find keypoints that appear in the "four" cameras
        _, _, matches_left_prev_curr = process_pair.get_keypoints_and_matches(prev.img_l, curr.img_l, rectified=False)
        fours_indexes = get_4matched_kps(matches_left_prev_curr, prev.matches, curr.matches)

        # convert fours from indexes of all kps, to the indexes of only the kps that where matched
        fours_indexes = [[prev.matched_indeces_kp_l[f[0]], prev.matched_indeces_kp_r[f[1]], curr.matched_indeces_kp_l[f[2]], curr.matched_indeces_kp_r[f[3]]] for f in fours_indexes]

        # Apply PnP using ransac
        # creates kps (2d pixel locations) of the 4-matched keypoints
        fours_kps = [[prev.kps_l[f[0]], prev.kps_r[f[1]], curr.kps_l[f[2]], curr.kps_r[f[3]]]
                     for f in fours_indexes]  # take the actual kps (not the indexes) for the four cameras


        # creates the corresponding 3d points, in world (left0) coordinates system TODO: check if to use prev_left coords, such that we get relative coords - from prev to curr
        kps_prev_left, kps_prev_right = np.array([p[0] for p in fours_kps]).T, np.array([p[1] for p in fours_kps]).T
        points_4d_world_hom = cv2.triangulatePoints(intrinsic @ prev.extrinsic_left,  # calibration left
                                              intrinsic @ prev.extrinsic_right,   # calibration right
                                              kps_prev_left,
                                              kps_prev_right)
        points_3d_world = points_4d_world_hom[:3, :] / points_4d_world_hom[3, :]  # vectors in the columns

        # find Rt for the curr pair
        R_curr_left, t_curr_left, is_supporters = get_Rt_with_ransac(points_3d_world.T, fours_kps, intrinsic, prev.extrinsic_left, prev.extrinsic_right)
        curr.extrinsic_left = hstack(R_curr_left, t_curr_left)
        curr.extrinsic_right = hstack(*get_R_t_right(R_curr_left, t_curr_left))

        # find/create the tracks in the pairs
        for i_fours in range(len(fours_indexes)):
            if is_supporters[i_fours]:
                curr_four = fours_indexes[i_fours]
                # check if the track is already exist in the prev pair
                if curr_four[0] in prev.matchIndex_TrackId:  # the track exists in previous pair
                    trackId = prev.matchIndex_TrackId[curr_four[0]]  # curr_four[0] == curr_four[1], because it goes as the match number
                    track = tracks[trackId]
                else:
                    track = Track()
                    # add the prev Pair to the track
                    track.PairId_MatchIndex[prev.PairId] = curr_four[0]
                    # add the track to the prev Pair
                    prev.matchIndex_TrackId[curr_four[0]] = track.TrackId

                    tracks[track.TrackId] = track

                # add the new Pair to the track
                track.PairId_MatchIndex[curr.PairId] = curr_four[2]
                # add the track to the new Pair
                curr.matchIndex_TrackId[curr_four[2]] = track.TrackId

        pairs[curr.PairId] = curr

        if i_pair % 500 == 0:
            database.serialize(file_path='data.pkl')

    database.serialize(file_path='data.pkl')

    show_camera_coords(database)

    """
    # have pair 0 and pair 1
    left0, right0 = part1.read_images(985)
    left1, right1 = part1.read_images(986)

    # ------------- find 4 keypoints that appear in the "four" cameras ----------------
    kps_prev_left, kps_prev_right, matches_pair0 = process_pair.get_keypoints_and_matches(left0, right0)
    kps_left1, kps_right1, matches_pair1 = process_pair.get_keypoints_and_matches(left1, right1)
    _, _, matches_left_0_1 = process_pair.get_keypoints_and_matches(left0, left1, rectified=False)

    fours_indexes = get_4matched_kps(matches_left_0_1, matches_pair0, matches_pair1)

    print("matches pair 0: ", len(matches_pair0))
    print("matches pair 1: ", len(matches_pair1))
    print("matches left 0-1: ", len(matches_left_0_1))
    print("matched kps on all 4 cameras:", len(fours_indexes))

    # --------------- Apply PnP using ransac ----------------
    # creates kps (2d pixel locations) of the 4-matched keypoints
    fours_kps = [[kps_prev_left[f[0]].pt, kps_prev_right[f[1]].pt, kps_left1[f[2]].pt, kps_right1[f[3]].pt]
                 for f in fours_indexes]  # take the actual kps (not the indexes) for the four cameras

    # creates the corresponding 3d points, in world (left0) coordinates system
    kps_prev_left, kps_prev_right = np.array([p[0] for p in fours_kps]).T, np.array([p[1] for p in fours_kps]).T
    k, m1, m2 = part1.read_cameras()
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, kps_prev_left, kps_prev_right)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]  # vectors in the columns

    R_left1, t_left1, is_supporters = get_Rt_with_ransac(points_3d_hom.T, fours_kps, k, m1, m2)

    # ------------ show position of the 4 cameras -------------
    cam_positions = get_cameras_locations(m1, m2, R_left1, t_left1)
    part1.plot_3d_points(cam_positions, zlim=[-1,2], xlim=[-1,2], ylim=[-1,2])

    # ------------ check supporters ---------------

    get_supporters(fours_kps, points_3d_hom, R_left1, t_left1, k, m1, m2, img_left0=left0, img_left1=left1)
    """


def show_camera_coords(database=None):
    if database:
        locations = []
        for i in range(len(database.Pairs)):
            pair = database.Pairs[i]
            locations.append(get_position_in_world(np.array([0,0,0]),  *extrinsic_to_Rt(pair.extrinsic_left)))

        # I'm not interested in y_axis, because it represents the height of the camera
        locations_0_2 = np.array([[l[0], l[2]] for l in locations])

        x, y = locations_0_2.T

        plt.scatter(x,y, c='red')  # s=np.array([5] * len(x)),

    # find the ground truth
    locations_ground_truth = []

    with open(GROUND_TRUTH_LOCATIONS_PATH, 'r') as f:
        for line in f:
            extrinsic = np.array([float(n) for n in line.split(' ')])
            extrinsic = np.reshape(extrinsic, (3,4))
            camera_in_camera_coords = np.array([0,0,0,1])
            locations_ground_truth.append(extrinsic @ camera_in_camera_coords)
    locations_ground_truth_0_2 = np.array([[l[0], l[2]] for l in locations_ground_truth])

    x, y = locations_ground_truth_0_2.T
    plt.scatter(x, y, c='blue')  # s=np.array([5] * len(x)),

    plt.legend((['predicted', 'ground truth'] if database else ['ground truth']))

    plt.show()


def hstack(R, t):
    len_shape_R = len(R.shape)
    len_shape_t = len(t.shape)

    if len(t.shape) < len_shape_R:
        t = np.reshape(t, t.shape + (1,)*(len_shape_R - len_shape_t))
    else:
        R = np.reshape(R, R.shape + (1,)*(len_shape_t - len_shape_R))

    return np.hstack((R, t))


def get_supporters(fours_kps, points_3d, R_left, t_left, intrinsic, extrinsic_left0, extrinsic_right0, img_left0=None, img_left1=None):
    #intrinsic, extrinsic_left0, extrinsic_right0 = part1.read_cameras()
    R_right, t_right = get_R_t_right(R_left, t_left)

    # [calib_left0, calib_right0, calib_left1, calib_right1]
    extrinsic_matrices = [extrinsic_to_Rt(extrinsic_left0),
                          extrinsic_to_Rt(extrinsic_right0),
                          (R_left, t_left),
                          (R_right, t_right)]

    is_supporter = []
    for (i, p_3d) in enumerate(points_3d.T):

        inlier = True
        for cam_idx in range(4):
            R, t = extrinsic_matrices[cam_idx]
            estimated_pixel_location = intrinsic @ (R @ p_3d + t)  # (extrinsic_matrices[cam_idx] @ np.reshape(np.append(p_3d, 1), (4, 1)))
            estimated_pixel_location = estimated_pixel_location[:2] / estimated_pixel_location[2]
            if np.linalg.norm(np.array(fours_kps[i][cam_idx]) - np.reshape(estimated_pixel_location, (2,))) >= THRESHOLD:
                is_supporter += [False]
                inlier = False
                break
        if inlier:
            is_supporter += [True]

    print("amount of supporters: ", sum(is_supporter), " out of ", len(is_supporter))

    if img_left0 is not None and img_left1 is not None:
        garbage_output = None
        for i in range(len(is_supporter)):
            if is_supporter[i]:
                # print in CYAN
                img_left0 = cv2.circle(img_left0, (int(fours_kps[i][0][0]), int(fours_kps[i][0][1])), radius=2, color=GREEN, thickness=-1)
                img_left1 = cv2.circle(img_left1, (int(fours_kps[i][2][0]), int(fours_kps[i][2][1])), radius=2, color=GREEN, thickness=-1)
            else:
                # print in ORANGE
                img_left0 = cv2.circle(img_left0, (int(fours_kps[i][0][0]), int(fours_kps[i][0][1])), radius=2, color=RED, thickness=-1)
                img_left1 = cv2.circle(img_left1, (int(fours_kps[i][2][0]), int(fours_kps[i][2][1])), radius=2, color=RED, thickness=-1)

        cv2.imshow("left 0 with inliers and outliers", img_left0)
        cv2.imshow("left 1 with inliers and outliers", img_left1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print(f"best amount of supporters: {sum(is_supporter)} / {len(is_supporter)}")

    return sum(is_supporter)/len(points_3d.T), is_supporter


def get_Rt_with_ransac(points_3d, kps_cameras, intrinsic, extrinsic_left0, extrinsic_right0):
    """
    get [R|t] of the camera using ransac with PnP as inner model
    steps:
    0. calculate amount of iterations
    repeat:
        1. select sample data
        2. create model
        3. check how the model fits to the data
        4. save best result
    5. refine transformation according to the inliers found in the loop
    :param points_3d: points in 3d, represented in world coordinate system
    :param kps_cameras: the fitting 2d points of the points_3d, for each camera
    :return: [R|t] of the second pair (the first one is known)
    """
    # calculate amount of iterations
    epsilon = 0.4  # primarily assumption, to be updated after
    iter = get_amount_of_iterations(prob=0.9999, sample_size=4, epsilon=epsilon)
    best_supporters_percentage = 0

    i_iteration = 0
    while i_iteration < max(5, iter):  # TODO: maybe get it smaller
        # select sample data
        index_samples = sorted(random.sample(range(len(points_3d)), 4))
        points_3d_sample = np.array([points_3d[idx] for idx in index_samples])
        kps_cameras_sample = np.array([kps_cameras[idx] for idx in index_samples])

        # calculate [R|t] using this sample
        extrinsic_left1 = apply_P3P(points_3d_sample.T, points_2d=np.array([f[2] for f in kps_cameras_sample]).T, intrinsic=intrinsic)

        # estimate the results
        if extrinsic_left1 is None:
            print("camera [R|t] not found")
            i_iteration -= 1  # TODO: check if that's the right thing, I just not trust it as a valid sample
            supporters_percentage = 0
        else:
            R_left1, t_left1 = extrinsic_left1
            supporters_percentage, supporters_boolean = get_supporters(kps_cameras, points_3d.T, R_left1, t_left1, intrinsic, extrinsic_left0, extrinsic_right0)

        if supporters_percentage > best_supporters_percentage:
            best_supporters_percentage = supporters_percentage
            best_supporters_boolean = supporters_boolean  # TODO: wasn't exist, check the meaning
            best_R, best_t = R_left1, t_left1

        if best_supporters_percentage == 1:
            print("found best result after", i_iteration, "iterations")
            break

        # update the amount of iterations
        i_iteration += 1
        epsilon = min(1 - best_supporters_percentage, 0.4)
        iter = get_amount_of_iterations(prob=0.9999, sample_size=4, epsilon=epsilon)

    if best_supporters_percentage == 0:
        return None, None, None

    # refine transformation according to the inliers found in the loop
    points_3d_pair0_inliers = np.array([points_3d[i] for i in range(len(points_3d)) if best_supporters_boolean[i]])
    kps_left1_inliers = np.array([kps_cameras[i][2] for i in range(len(kps_cameras)) if best_supporters_boolean[i]])

    ret, rvecs, tvecs = cv2.solvePnP(
        objectPoints=points_3d_pair0_inliers,
        imagePoints=kps_left1_inliers,
        cameraMatrix=intrinsic,
        distCoeffs=None)

    R_PnP, _ = cv2.Rodrigues(np.reshape(rvecs, (3,)))  # convert rotation vector to rotation matrix
    t_PnP = np.reshape(tvecs, (3,))

    refined_supporters_percentage, refined_supporters_boolean = get_supporters(kps_cameras, points_3d.T, R_PnP, t_PnP, intrinsic, extrinsic_left0, extrinsic_right0)

    if refined_supporters_percentage < best_supporters_percentage:
        print(f"refined is worse than the best P3P. refined - {sum(refined_supporters_boolean)}, best P3P - {sum(best_supporters_boolean)}")
        R, t = best_R, best_t
        supporters_boolean = best_supporters_boolean
    else:
        R, t = R_PnP, t_PnP
        supporters_boolean = refined_supporters_boolean
    """
    # always take the PnP
    R, t = R_PnP, t_PnP
    supporters_boolean = refined_supporters_boolean
    """
    print(f"best amount of supporters: {sum(supporters_boolean)} / {len(supporters_boolean)}")

    return R, t, supporters_boolean


def get_amount_of_iterations(prob, sample_size, epsilon):
    return ceil(log2(1-prob) / log2(1 - (1 - epsilon)**sample_size))


def get_cameras_locations(m1, m2, R_left1, t_left1):
    position_in_cam_coords = np.array([0, 0, 0])
    left0_position = get_position_in_world(position_in_cam_coords, *extrinsic_to_Rt(m1))
    right0_position = get_position_in_world(position_in_cam_coords, *extrinsic_to_Rt(m2))
    left1_position = get_position_in_world(position_in_cam_coords, R_left1, t_left1)
    right1_position = get_position_in_world(position_in_cam_coords, *get_R_t_right(R_left1, t_left1))
    cam_positions = np.hstack((np.reshape(left0_position, (3, 1)), np.reshape(right0_position, (3, 1))))
    cam_positions = np.hstack((cam_positions, np.reshape(left1_position, (3, 1))))
    cam_positions = np.hstack((cam_positions, np.reshape(right1_position, (3, 1))))
    return cam_positions


def get_position_in_world(point, R, t):
    """
    change the coordinate system of point. Camera coords --> world coords.
    :param point: the point to change coords to. starts in camera coords
    :param R: rotation from world to camera
    :param t: translation from world to camera
            i.e. p_camera = R * p_world + t
    :return: the point in world coordinates
    """
    t = np.reshape(t, (3,))
    return R.T @ (point - t)


def get_R_t_right(R_left, t_left):

    t_left = np.reshape(t_left, (3,))
    _, _, m2 = part1.read_cameras()
    R_lr, t_lr = extrinsic_to_Rt(m2)

    R = R_lr @ R_left
    t = R_lr @ t_left + t_lr

    return R, t


def extrinsic_to_Rt(extrinsic):
    return extrinsic[:, :3], np.reshape(extrinsic[:, 3], (3,))


def apply_P3P(points_3d, points_2d, intrinsic):
    """
    finds the Rotation and translation matrix of the camera, from world coords
    :param points_3d: 4 3d_points in world coords-sys. (Note: the vectors are in the columns)
    :param points_2d: 4 matching pixel locations in the camera image plane (Note: the vectors are in the columns)
    :param intrinsic: intrinsic matrix of new camera
    :return: [R|t] of new camera. meaning: p(camera coords) = [R|t] * W(world coords)
    """
    points_3d = np.ascontiguousarray(points_3d.T)  # the vectors are now in rows
    points_2d = np.ascontiguousarray(points_2d.T)  # the vectors are now in rows

    objectPoints = points_3d[:3]  # takes 3 points
    image_points = np.ascontiguousarray(points_2d[:3])

    retval, rvec, tvec = cv2.solveP3P(objectPoints=objectPoints,
                                      imagePoints=image_points,
                                      cameraMatrix=intrinsic,
                                      distCoeffs=None,
                                      flags=cv2.SOLVEPNP_P3P)

    # check the returned values from solveP3P
    validation_vec = points_3d[3]  # takes the fourth vector, to check which option is the current one
    validation_2d = points_2d[3]

    for i in range(retval):
        R, _ = cv2.Rodrigues(rvec[i])  # convert rotation vector to rotation matrix
        t = np.reshape(tvec[i], (3,))

        estimated_2d_hom = intrinsic @ (R @ validation_vec + t)
        estimated_2d = estimated_2d_hom[:2] / estimated_2d_hom[2]
        if np.linalg.norm(validation_2d - estimated_2d) < 1:  # if 2d == k*[R|t]*vec (almost equal is good enough)
            # print("found the extrinsic matrix of the camera!\n the possible values: ", retval)
            return R, t

    return None


def get_4matched_kps(matches_left_0_1, matches_pair0, matches_pair1):
    """
    find quartets of indeces of the same keypoint, as recognized by the 4 cameras
    :param matches_left_0_1: matches between left0 and left1
    :param matches_pair0: matches between left0 and right0
    :param matches_pair1: matches between left1 and right1
    :return: quartets of indeces of the same keypoint, as recognized by the 4 cameras.
             the order of returned indeces is [left0, right0, left1, right1]
    """
    # Reminder: img1 = queryIdx, img2 = trainIdx
    fours = []
    for m0 in matches_pair0:
        right0_match = m0.trainIdx
        left0_match = m0.queryIdx
        for m_left in matches_left_0_1:
            if m_left.queryIdx == left0_match:
                left1_match = m_left.trainIdx
                for m1 in matches_pair1:
                    if m1.queryIdx == left1_match:
                        right1_match = m1.trainIdx
                        fours.append([left0_match, right0_match, left1_match, right1_match])
                        break
                break
    return fours


if __name__ == "__main__":
    main()
