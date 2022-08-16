import part1
import process_pair
import random
import cv2
import numpy as np
import itertools

THRESHOLD = 1.5  # 2
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255,255,0)
ORANGE = (0, 69, 255)


class Track:
    new_id = itertools.count()

    def __init__(self):
        self.TrackId = next(Track.new_id)


class Pair:
    def __init__(self, pair_id: int, img_l, img_r):
        self.PairId = pair_id
        self.img_l = img_l
        self.img_r = img_r

        self.kps_l, self.kps_r, self.matches = process_pair.get_keypoints_and_matches(img_l, img_r)

        matched_kps_l, matched_kps_r = part1.get_matched_kps_from_matches(self.matches, self.kps_l, self.kps_r)
        self.matched_kps_l = np.array([kp.pt for kp in matched_kps_l])
        self.matched_kps_r = np.array([kp.pt for kp in matched_kps_r])

        self.cloud = process_pair.get_cloud(matched_kps_l, matched_kps_r,
                                            *part1.read_cameras())
        self.R = None
        self.t = None


def main():
    # have pair 0 and pair 1
    left0, right0 = part1.read_images(0)
    left1, right1 = part1.read_images(1)

    # ------------- find 4 keypoints that appear in the "four" cameras ----------------
    kps_left0, kps_right0, matches_pair0 = process_pair.get_keypoints_and_matches(left0, right0)
    kps_left1, kps_right1, matches_pair1 = process_pair.get_keypoints_and_matches(left1, right1)
    _, _, matches_left_0_1 = process_pair.get_keypoints_and_matches(left0, left1)

    fours_indeces = get_4matched_kps(matches_left_0_1, matches_pair0, matches_pair1)

    print("matches pair 0: ", len(matches_pair0))
    print("matches pair 1: ", len(matches_pair1))
    print("matches left 0-1: ", len(matches_left_0_1))
    print("matched kps on all 4 cameras:", len(fours_indeces))

    # --------------- Apply PnP ----------------
    fours_to_PnP = random.sample(fours_indeces, 4)

    # create 3d points from pair0
    pair0_2d = [[kps_left0[f[0]], kps_right0[f[1]]] for f in fours_to_PnP]  # take the actual kps (not the indeces) for left0 and right0
    kps_left0_PnP, kps_right0_PnP = np.array([p[0].pt for p in pair0_2d]).T, np.array([p[1].pt for p in pair0_2d]).T
    k, m1, m2 = part1.read_cameras()
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, kps_left0_PnP, kps_right0_PnP)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]  # vectors in the columns

    part1.plot_3d_points(points_3d_hom)  # plot the four 3d_points

    # get 2d points of left1
    left1_2d = np.array([kps_left1[f[2]].pt for f in fours_to_PnP])
    left1_2d = left1_2d.T

    extrinsic_left1 = apply_P3P(points_3d_hom, left1_2d, k)  # given W in left0 coords, extrinsic_left1 @ W will be the same point in left1 coords
    R_left1, t_left1 = None, None
    if extrinsic_left1 is not None:
        R_left1, t_left1 = extrinsic_left1
    else:
        print("camera [R|t] not found")
        return

    # ------------ show position of the 4 cameras -------------
    cam_positions = get_cameras_locations(m1, m2, R_left1, t_left1)
    part1.plot_3d_points(cam_positions, zlim=[-1,2], xlim=[-1,2], ylim=[-1,2])

    # ------------ check supporters ---------------
    fours_kps = [[kps_left0[f[0]].pt, kps_right0[f[1]].pt, kps_left1[f[2]].pt, kps_right1[f[3]].pt]
                for f in fours_indeces]  # take the actual kps (not the indeces) for the four cameras
    kps_left0, kps_right0 = np.array([p[0] for p in fours_kps]).T, np.array([p[1] for p in fours_kps]).T
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, kps_left0, kps_right0)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]  # vectors in the columns

    get_supporters(fours_kps, points_3d_hom, R_left1, t_left1, img_left0=left0, img_left1=left1)


def get_supporters(fours_kps, points_3d, R_left, t_left, img_left0=None, img_left1=None):
    k, m1, m2 = part1.read_cameras()
    R_right, t_right = get_R_t_right(R_left, t_left)

    # [calib_left0, calib_right0, calib_left1, calib_right1]
    extrinsic_matrices = [extrinsic_to_Rt(m1),
                          extrinsic_to_Rt(m2),
                          (R_left, t_left),
                          (R_right, t_right)]

    is_supporter = []
    for (i, p_3d) in enumerate(points_3d.T):

        inlier = True
        for cam_idx in range(4):
            R, t = extrinsic_matrices[cam_idx]
            estimated_pixel_location = k @ (R @ p_3d + t)  # (extrinsic_matrices[cam_idx] @ np.reshape(np.append(p_3d, 1), (4, 1)))
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

    return len(is_supporter)/len(points_3d), is_supporter


def get_Rt_with_ransac(points_3d, kps_cameras, intrinsic, extrinsic_left0, extrinsic_right0):
    """
    get [R|t] of the camera using ransac with PnP as inner model
    repeat:
    1. select sample data
    2. create model
    3. check how the model fits to the data
    4. save best result
    :param points_3d: points in 3d, represented in world coordinate system
    :param kps_cameras: the fitting 2d points of the points_3d, for each camera
    :return: [R|t] of the second pair (the first one is known
    """
    best_percentage = 0

    for i_iteration in range(100):  # todo: create a good condition, using statistics
        # select sample data
        index_samples = sorted(random.sample(range(len(points_3d)), 4))
        points_3d_sample = [points_3d[idx] for idx in index_samples]
        kps_cameras_sample = [kps_cameras[idx] for idx in index_samples]

        # calculate [R|t] using this sample
        extrinsic_left1 = apply_P3P(points_3d_sample, points_2d=[f[2] for f in kps_cameras_sample], intrinsic=intrinsic)

        # estimate the results
        supporters_percentage = 0
        if extrinsic_left1 is None:
            print("camera [R|t] not found")
            supporters_percentage = 0
        else:
            R_left1, t_left1 = extrinsic_left1
            supporters_percentage, supporters_boolean = get_supporters(kps_cameras, points_3d, R_left1, t_left1)

        if supporters_percentage > best_percentage:
            best_percentage = supporters_percentage
            best_R, best_t = R_left1, t_left1

        if best_percentage == 1:
            return best_R, best_t

    return (best_R, best_t) if best_percentage > 0 else (None, None)


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
            p_camera = R * p_world + t
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
            print("found the extrinsic matrix of the camera!\n the possible values: ", retval)
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
