import copy
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gtsam

import part1
import part2
import part3
import process_pair
from part2 import Database, Pair, Track, demi_match

SOURCE_DATA = r'results/data 05.pkl'
CYAN = (255,255,0)
RED = (0, 0, 255)
POSE_UNCERTAINTY = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))




def main():
    """
    This file checks the performance of bundle adjustment.
    1. Visualizes the track in 3d, with the location of the camera.
    2. Measures the accumulated error through the progress of the pairs.
    3. Present the accumulated error through the pairs visually, in the images (with comparison to the detection).
    4. Solve bundle adjustment, and shows the camera locations in ground truth, initial estimation and after the bundle
    5. Check performance after bundle: numeric distance from the detected feature, numeric distance from ground truth extrinsic matrices
    6. visualizes the 2d location, with comparison to the detected feature (same as step 3)
    """
    database_path = SOURCE_DATA 
    database = Database(database_path)

    if (type(database.Pairs) is dict):
        database.Pairs = list(database.Pairs.values())
    
    track = database.Tracks[22844]

    # get the 3d location of the track, in coords of the first camera in bundle
    intrinsic, _, m2 = part1.read_cameras()
    points3d, point3d = get_3d_location(database, track, intrinsic)
    pair = database.Pairs[list(track.PairId_MatchIndex.keys())[0]]
    camera_location = part2.get_position_in_world(np.array([0,0,0]), *part2.get_Rt(pair.extrinsic_left))
    
    # print the cloud, with the specific point3d and the camera location
    plot_3d_points(points3d, point3d, camera_location)

    dis_left, dis_right, error = [], [], []

    for pair_id in track.PairId_MatchIndex:
        # detected location
        detected_l, detected_r = database.feature_location(pair_id, track.TrackId)
        # expected location, from projection of the point3d
        projected_l, projected_r = get_projected_location(database.Pairs[pair_id], intrinsic, point3d)

        dis_left.append(math.sqrt((detected_l[0]-projected_l[0])**2 + (detected_l[1]-projected_l[1])**2)) 
        dis_right.append(math.sqrt((detected_r[0]-projected_r[0])**2 + (detected_r[1]-projected_r[1])**2)) 
        mean_detected_y, mean_projected_y = (detected_l[1] + detected_r[1])/2, (projected_l[1] + projected_r[1])/2
        error.append(0.5*((detected_l[0]-projected_l[0])**2 + 
                        (detected_r[0]-projected_r[0])**2 +
                        (mean_detected_y-mean_projected_y)**2))

    plt.clf()
    plt.plot(dis_left)
    plt.grid()
    plt.title("distances in left camera")
    plt.show()

    plt.plot(dis_right)
    plt.grid()
    plt.title("distances in right camera")
    plt.show()

    plt.cla()
    plt.plot(error)
    plt.grid()
    plt.title("errors from cameras")
    plt.show()

    show_track_in_images(track, database, point3d)

    bundle = copy.deepcopy(database.Pairs[list(track.PairId_MatchIndex.keys())[0]:list(track.PairId_MatchIndex.keys())[-1] + 1])
    part3.positioning_bundle(bundle, np.identity(4)[:3, :])

    plt.clf()
    # plot ground truth
    gt_extrinsics = get_ground_truth_extrinsics_partial(bundle[0].PairId, bundle[-1].PairId)  # [bundle[0].PairId:bundle[-1].PairId+1]
    part3.positioning_extrinsics(gt_extrinsics, np.identity(4)[:3, :])
    x, y = part2.get_extrinsics_locations(gt_extrinsics).T
    plt.scatter(x, y, c='blue', s=np.array([5] * len(x)))
    
    # plot the old location
    locations = part2.get_extrinsics_locations([b.extrinsic_left for b in bundle])
    x, y = locations.T
    plt.scatter(x, y, c='green', s=np.array([5] * len(x)))
    
    bundled, track_loc = solve_bundle(copy.deepcopy(bundle))
    
    # plot the bundled pairs
    locations = part2.get_extrinsics_locations([b.extrinsic_left for b in bundled])
    x, y = locations.T
    plt.scatter(x, y, c='orange', s=np.array([5] * len(x)))
    
    plt.legend(['ground truth', 'intial', 'bundled'])
    plt.show()

    # calculates the error after the bundle
    error = []
    i = 0
    for pair_id in track.PairId_MatchIndex:
        # detected location
        detected_l, detected_r = database.feature_location(pair_id, track.TrackId)
        # expected location, from projection of the point3d
        projected_l, projected_r = get_projected_location(bundled[i], intrinsic, track_loc)  # track_loc instead of point3d

        mean_detected_y, mean_projected_y = (detected_l[1] + detected_r[1])/2, (projected_l[1] + projected_r[1])/2
        error.append(0.5*((detected_l[0]-projected_l[0])**2 + 
                        (detected_r[0]-projected_r[0])**2 +
                        (mean_detected_y-mean_projected_y)**2))
        i+=1

    plt.plot(error)
    plt.grid()
    plt.title("errors from bundle")
    plt.show()

    # check performance by similarity to ground truth extrinsic matrices
    eval_performance([b.extrinsic_left for b in bundled], gt_extrinsics, "R error of bundled")

    # # Optional: returning the track location and the bundled to global coords 
    # first_pair = database.Pairs[list(track.PairId_MatchIndex.keys())[0]]
    # R, t = part2.get_Rt(first_pair.extrinsic_left)
    # R, t = part3.get_inverted_transformation(R, t)
    # track_loc = R @ track_loc + t

    # part3.positioning_bundle(bundled, database.Pairs[110].extrinsic_left)
    
    for (i, p) in enumerate(database.Pairs[110:143]):
        p.extrinsic_left = bundled[i].extrinsic_left

    show_track_in_images(track, database, track_loc)
    

def eval_performance(extrinsics, gt_extrinsics, title):
    R_err = 0
    R_values = []

    for i in range(len(extrinsics)-1):
        # finding relative transformations of extrinsic and ground truth matrices
        rel_ext = part3.get_relative_transformation(extrinsics[i], extrinsics[i+1])
        rel_gt_ext = part3.get_relative_transformation(gt_extrinsics[i], gt_extrinsics[i+1])

        # finding the difference between the two
        rel_ex = part3.get_relative_transformation(rel_gt_ext, rel_ext)

        # rel_ex = part3.get_relative_transformation(gt_extrinsics[i], extrinsics[i])
        R, _ = part2.get_Rt(rel_ex)
        temp, _ = cv2.Rodrigues(R)
        R_err += np.dot(temp.T, temp)[0][0]
        R_values.append(np.dot(temp.T, temp)[0][0])

    plt.plot(R_values)
    plt.title(title)
    plt.show()

    return R_err


def solve_bundle(bundle):
    """
    perform bundle adjustment on the bundle. refine locations of cameras and keypoints

    :param bundle: the bundle for the adjustment
    :param tracks: tracks of all the pairs
    :param const_start: boolean value to determine whether the beginning location is unchangable
    :return: the bundle after the adjustment
    """

    # Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # create clibration (intrinsic) matrix
    intrinsic, _, m2 = part1.read_cameras()
    cameras_x_distance = -m2[0, 3]  # minus because the value should be positive
    # format: fx fy skew cx cy baseline 
    K = gtsam.Cal3_S2Stereo(fx=intrinsic[0,0], fy=intrinsic[1,1], s=intrinsic[0,1], u0=intrinsic[0,2], v0=intrinsic[1,2], b=cameras_x_distance)
    # declare the first camera
    camera0 = gtsam.symbol('c', 0)

    # give prior knowledge on the beginning position of the bundle- to be the end of last bundle
    R, t = part2.get_Rt(bundle[0].extrinsic_left)
    R, t = part3.get_inverted_transformation(R, t)  # inverting the transformation because GTSAM expects transformation from world to camera
    first_pose = gtsam.Pose3(gtsam.Rot3(R), t)  
    graph.add(gtsam.PriorFactorPose3(camera0, first_pose, POSE_UNCERTAINTY))
    #graph.add(gtsam.NonlinearEqualityPose3(camera0, first_pose))  # TODO: check if it's better - setting the starting position unchanged

    ## for each pair: add the points3d with factors
    # start with measurement noise model (of detector)
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
    
    for (i, pair) in enumerate(bundle):
        camera = gtsam.symbol('c', i)

        # add the factors from projecting points3d to image plane
        # Note: each point3d is identified uniquely by the trackId
        for (matchedIdx, trackId) in pair.matchIndex_TrackId.items():
            point3d = gtsam.symbol('p', trackId)
            (x_left, y_left), (x_right, y_right) = pair.kps_l[matchedIdx], pair.kps_r[matchedIdx]
            
            point_factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(x_left, x_right, (y_left+y_right)/2), stereo_model, camera, point3d, K)
            graph.add(point_factor)

    # fill initial estimate with cameras and points3d locations
    
    for i in range(len(bundle)):
        # insert all camera positions
        R, t = part2.get_Rt(bundle[i].extrinsic_left)
        R, t = part3.get_inverted_transformation(R, t)
        pose = gtsam.Pose3(gtsam.Rot3(R), t)
        if not initial_estimate.exists(gtsam.symbol('c', i)):  # TODO: is it redundant?
            initial_estimate.insert(gtsam.symbol('c', i), pose)

        # create points3d cloud
        extrinsic_left = bundle[i].extrinsic_left
        extrinsic_right = part2.hstack(*part2.get_Rt_right(*part2.get_Rt(extrinsic_left)))
        points3d = process_pair.get_cloud(bundle[i].kps_l, bundle[i].kps_r, intrinsic, extrinsic_left, extrinsic_right).T  # points3d are in world coordinates
        
        # invert the points3d location, to coordinates in the standard of gtsam
        # R, t = part3.get_inverted_transformation(*part2.get_Rt(bundle[i].extrinsic_left))
        # points3d = (R @ points3d.T).T + t # TODO: check, added


        # enter initials to every point that wasn't yet given an intial value
        for (matchIdx, trackId) in bundle[i].matchIndex_TrackId.items():
            curr_point3d = gtsam.symbol('p', trackId)
            # add if the beginning of bundle or beginning of track
            if not initial_estimate.exists(curr_point3d):
                # inverted = gtsam.Point3(R @ points3d[matchIdx] + t)
                initial_estimate.insert(curr_point3d, points3d[matchIdx])
    
    # optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # get results
    for i in range(len(bundle)):
        pose = result.atPose3(gtsam.symbol('c', i))
        R = pose.rotation().matrix()
        t = np.array([pose.x(), pose.y(), pose.z()])
        R, t = part3.get_inverted_transformation(R, t)
        bundle[i].extrinsic_left = part2.hstack(R, t)

    track = result.atPoint3(gtsam.symbol('p', 22844))

    return bundle, track


def get_ground_truth_extrinsics_partial(start_idx, end_idx):
    # find the ground truth that matches the pairs
    extrinsics = []
    i = 0
    with open(part2.GROUND_TRUTH_LOCATIONS_PATH, 'r') as f:
        for line in f:
            if i < start_idx:
                i += 1
            elif i <= end_idx:
                # read the extrinsic data from the file
                extrinsic = np.array([float(n) for n in line.split(' ')])
                extrinsic = np.reshape(extrinsic, (3,4))
                # invert the homography, to be [R|t] @ [p_w|1] = [p_c], according to the standard we work with
                gt_R, gt_t = extrinsic[:, :3], extrinsic[:, 3]
                gt_R, gt_t = part3.get_inverted_transformation(gt_R, gt_t)
                extrinsics.append(part2.hstack(gt_R, gt_t))
                i += 1
            else:
                break

    return np.array(extrinsics)


def get_projected_location(pair, intrinsic, point3d):
    """
    projecting the 3d point back to the image plane of the pair
    """
    R_left, t_left = part2.get_Rt(pair.extrinsic_left)
    R_right, t_right = part2.get_Rt_right(R_left, t_left)
    expected_l = intrinsic @ (R_left @ point3d + t_left)
    expected_l = expected_l[:2] / expected_l[2]
    expected_r = intrinsic @ (R_right @ point3d + t_right)
    expected_r = expected_r[:2] / expected_r[2]
    return expected_l,expected_r

def get_3d_location(database, track, intrinsic):
    """
    find 3d location of track, generated by the first camera it was found in
    """
    pair_id = list(track.PairId_MatchIndex.keys())[0]
    pair = database.Pairs[pair_id]
    
    # create points3d cloud
    extrinsic_left = pair.extrinsic_left
    extrinsic_right = part2.hstack(*part2.get_Rt_right(*part2.get_Rt(extrinsic_left)))
    points3d = process_pair.get_cloud(pair.kps_l, pair.kps_r, intrinsic, extrinsic_left, extrinsic_right)  # points3d are in world coordinates
    
    # get location of specific point3d of the track
    point3d = points3d.T[track.PairId_MatchIndex[pair_id]]
    return points3d,point3d


def get_tracks_in_range(tracks, start_pair, end_pair, min_len = 1):
    """
    return found tracks in the pairs range, when track length >= min_len
    """
    in_range = []
    for t in tracks.values():
        if start_pair <= list(t.PairId_MatchIndex.keys())[0] <end_pair and len(t.PairId_MatchIndex) >= min_len:
            in_range.append(t)

    in_range.sort(reverse=True, key= lambda t: len(t.PairId_MatchIndex))
    
    return in_range


def show_track_in_images(track, database, track3d=None):
    """
    showing track in the images, as found by the detector. 
    If an approximation of the track location in 3d was given - project it too to the images, and see the difference  
    """
    for pair_id in track.PairId_MatchIndex:
        kp_l, kp_r = database.feature_location(pair_id, track.TrackId)
        img_l, img_r = part1.read_images(pair_id)
        kped_img_l = cv2.circle(img_l, (int(kp_l[0]), int(kp_l[1])), 4, color=CYAN, thickness=-1)
        kped_img_r = cv2.circle(img_r, (int(kp_r[0]), int(kp_r[1])), 4, color=CYAN, thickness=-1)
        
        if not track3d is None:
            intrinsic = part1.read_cameras()[0]
            projected_l, projected_r = get_projected_location(database.Pairs[pair_id], intrinsic, track3d)
            kped_img_l = cv2.circle(img_l, (int(projected_l[0]), int(projected_l[1])), 4, color=RED, thickness=-1)
            kped_img_r = cv2.circle(img_r, (int(projected_r[0]), int(projected_r[1])), 4, color=RED, thickness=-1)

        cv2.imshow(f"follow track left image", kped_img_l)
        # cv2.imshow(f"follow track right image", kped_img_r)
        cv2.waitKey(250)


def plot_3d_points(points_3d, point3d, cam_loc=None, zlim=[], xlim=[], ylim=[]):
    # plotting the triangulation
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # limiting the values of x,y axes to achieve "zoom in" (not necessary,
    ax.set_zlim([-25, 275]) if zlim == [] else ax.set_zlim(zlim)
    plt.xlim([-30, 20]) if xlim == [] else plt.xlim(xlim)
    plt.ylim([-10, 2]) if ylim == [] else plt.ylim(ylim)
    # scatter points
    ax.scatter(points_3d[0, :], points_3d[1, :], zs=points_3d[2, :],
               s=20, c='b', depthshade=True, alpha=0.5)
    ax.scatter([point3d[0]], [point3d[1]], zs=[point3d[2]],
               s=200, c='r', depthshade=True, alpha=1)
    if not cam_loc is None:
        ax.scatter([cam_loc[0]], [cam_loc[1]], zs=[cam_loc[2]],
               s=200, c='g', depthshade=True, alpha=1)
    plt.show()


if __name__ == "__main__":
    main()
