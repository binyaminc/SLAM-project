import copy
import math
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import gtsam
from tqdm import tqdm

import part1
import part2
import process_pair
from part2 import Database, Pair, Track, demi_match

AVG_DIS = 50  # 7.3 average of distance passed in 10 frames
POSE_UNCERTAINTY = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))
DEBUG = True
SOURCE_DATA = r'results/data 01.pkl'


def main():

    database_path = SOURCE_DATA 
    database = Database(database_path)
    
    part2.show_camera_coords(database.Pairs)
    
    database.Pairs = solve_bundles(database.Pairs, database.Tracks)
    
    database.serialize(file_path=database_path[:-4] + f' - bundled {AVG_DIS}.pkl')
    plt.clf()
    part2.show_camera_coords(database.Pairs)
    input()
    

def split_data(full_database, start_idx, end_idx, save_path):
    """
    splitting the data. use example:
    
    split_data(database, 2476, 2524, 'results/sample data - 2476 to 2524.pkl')
    
    database = part2.Database()
    with open(database_path, 'rb') as f:
           database.Pairs, database.Tracks, starting_position = pickle.load(f)
    adjed_pairs = solve_bundles(database.Pairs, database.Tracks, sample=None, a_starting_position=starting_position)
    """
    ext_database = Database()
    ext_database.Pairs, ext_database.Tracks = extract_partial_path(list(full_database.Pairs.values()), full_database.Tracks, start_idx, end_idx)
    starting_position = solve_bundles(full_database.Pairs, full_database.Tracks, sample=start_idx)
    with open(save_path, 'wb') as f:
            pickle.dump((ext_database.Pairs, ext_database.Tracks, starting_position), f, pickle.HIGHEST_PROTOCOL)


def solve_bundles(pairs, tracks, sample=None, a_starting_position=None):
    """
    Solves bundle adjustment. steps:
    1. splits the pairs (frames) into small bundles according to distance
    2. solve each bundle 
    3. concatenate bundles and return
    """
    if type(pairs) is dict:
        pairs = list(pairs.values())

    last_end_x, last_end_y = 0, 0  # position of the end of last bundle
    seperating_idxes = [0]  # the indexes that seperate bundles. Each is the end of previous and start of new bundle
    starting_position = pairs[0].extrinsic_left.copy()  # starting position of new bundle
    last_extrinsic = pairs[0].extrinsic_left.copy()
    adjed_pairs = []

    if a_starting_position is not None:
        starting_position = a_starting_position

    for i in tqdm(range(len(pairs))):
        
        # finds the position of current pair and distance to end of last bundle
        (curr_x, _, curr_y) = part2.get_position_in_world(np.array([0, 0, 0]), *part2.get_Rt(pairs[i].extrinsic_left))
        dis = math.sqrt((curr_x - last_end_x) ** 2 + (curr_y - last_end_y) ** 2)
        if a_starting_position is not None:
            dis = 0 # for debug purposes

        if dis > AVG_DIS or i == len(pairs)-1:
   
            # start new bundle
            start_idx = seperating_idxes[-1]
            end_idx = i
            seperating_idxes.append(i)
            bundle = pairs[start_idx: end_idx + 1]
            print(f'\rbundle from in {start_idx} to {end_idx}  ', end="")
            
            if start_idx == sample:
                return starting_position
                #print("starting check")
            
            bundle[0].extrinsic_left = last_extrinsic  # the first extrinsic of the bundle should be from the old location, and not from the end of the last adjusted bundle
            
            if DEBUG:
                plt.clf()
                # plot ground truth
                ground_truth = part2.get_ground_truth_locations()[bundle[0].PairId:bundle[-1].PairId+1]
                x, y = ground_truth.T
                plt.scatter(x, y, c='blue', s=np.array([5] * len(x)))
            
                # plot the old location
                locations = part2.get_pairs_locations(bundle)
                x, y = locations.T
                plt.scatter(x, y, c='green', s=np.array([5] * len(x)))
                
            # save data of last pair in new bundle
            last_extrinsic = bundle[-1].extrinsic_left.copy()
            (last_end_x, _, last_end_y) = part2.get_position_in_world(np.array([0, 0, 0]), *part2.get_Rt(last_extrinsic))
            
            # positioning the bundle to start from the starting position
            positioning_bundle(bundle, starting_position)

            if DEBUG: 
                # plot the new location
                locations = part2.get_pairs_locations(bundle)
                x, y = locations.T
                plt.scatter(x, y, c='orange', s=np.array([5] * len(x)))
            
            adjed_bundle, starting_position = solve_bundle(bundle, tracks)

            if DEBUG:
                # plot the bundled location
                locations = part2.get_pairs_locations(bundle)
                x, y = locations.T
                plt.scatter(x, y, c='red', s=np.array([5] * len(x)))
                plt.legend(['ground truth', 'intial', 'positioned', 'bundled'])
                
                #plt.savefig(f'results/bundles/from_{start_idx}_to_{end_idx}.png')
                plt.show()

            adjed_pairs.extend(adjed_bundle)

    return adjed_pairs


def extrinsic_dis(ext1, ext2):
    """
    function to measure the distance between two extrinsice matrices of cameras
    """
    R1, t1 = part2.get_Rt(ext1)
    R2, t2 = part2.get_Rt(ext2)

    p_ext1 = part2.get_position_in_world(np.array([0,0,0]), R1, t1)  # position of extrinsic1
    p_ext2 = part2.get_position_in_world(np.array([0,0,0]), R2, t2)  # position of extrinsic2
    
    dis = math.sqrt((p_ext1[0] - p_ext2[0]) ** 2 + (p_ext1[2] - p_ext2[2]) ** 2)
    return dis


def solve_bundle(bundle, tracks):
    """
    perform bundle adjustment on the bundle. refine locations of cameras and keypoints

    :param bundle: the bundle for the adjustment
    :param tracks: tracks of all the pairs
    :param const_start: boolean value to determine whether the beginning location is unchangable
    :return: the bundle after the adjustment + last position
    """

    # Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # create clibration (intrinsic) matrix
    intrinsic, _, m2 = part1.read_cameras()
    cameras_x_distance = -m2[0, 3]  # minus because the value should be positive
    # format: fx fy skew cx cy baseline 
    K = gtsam.Cal3_S2Stereo(fx=intrinsic[0,0], fy=intrinsic[1,1], s=0.0, u0=intrinsic[0,2], v0=intrinsic[1,2], b=cameras_x_distance)

    # declare the first camera
    camera0 = gtsam.symbol('c', 0)

    # fix beginning position to be the end of last bundle
    R, t = part2.get_Rt(bundle[0].extrinsic_left)
    R, t = get_inverted_transformation(R, t)  # inverting the transformation because GTSAM expects transformation from world to camera
    first_pose = gtsam.Pose3(gtsam.Rot3(R), t)  
    graph.add(gtsam.PriorFactorPose3(camera0, first_pose, POSE_UNCERTAINTY))

    ## for each pair: add the points3d with factors
    # start with measurement noise model (of detector)
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))  # TODO: check if this is the sigma I want
    
    for (i, pair) in enumerate(bundle):
        camera = gtsam.symbol('c', i)

        # add the factors from projecting points3d to image plane
        # Note: each point3d is identified uniquely by the trackId
        for (matchedIdx, trackId) in pair.matchIndex_TrackId.items():
            point3d = gtsam.symbol('p', trackId)
            (x_left, y_left), (x_right, y_right) = pair.kps_l[matchedIdx], pair.kps_r[matchedIdx]
            
            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(x_left, x_right, (y_left+y_right)/2), stereo_model, camera, point3d, K))

    # fill initial estimate with cameras and points3d locations
    
    for i in range(len(bundle)):
        # insert all camera positions
        R, t = part2.get_Rt(bundle[i].extrinsic_left)
        R, t = get_inverted_transformation(R, t)
        pose = gtsam.Pose3(gtsam.Rot3(R), t)
        if not initial_estimate.exists(gtsam.symbol('c', i)):  # TODO: is it redundant?
            initial_estimate.insert(gtsam.symbol('c', i), pose)

        # create points3d cloud
        extrinsic_left = bundle[i].extrinsic_left
        extrinsic_right = part2.hstack(*part2.get_Rt_right(*part2.get_Rt(extrinsic_left)))
        points3d = process_pair.get_cloud(bundle[i].kps_l, bundle[i].kps_r, intrinsic, extrinsic_left, extrinsic_right).T  # points3d are in world coordinates
        
        # enter initials to every point that wasn't yet given an intial value
        for (matchIdx, trackId) in bundle[i].matchIndex_TrackId.items():
            curr_point3d = gtsam.symbol('p', trackId)
            # add if the beginning of bundle or beginning of track
            if not initial_estimate.exists(curr_point3d):
                initial_estimate.insert(curr_point3d, points3d[matchIdx])
    
    # optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # get results
    for i in range(len(bundle)):
        pose = result.atPose3(gtsam.symbol('c', i))
        R = pose.rotation().matrix()
        t = np.array([pose.x(), pose.y(), pose.z()])
        R, t = get_inverted_transformation(R, t)
        bundle[i].extrinsic_left = part2.hstack(R, t)
    
    next_starting_position = bundle[-1].extrinsic_left.copy()
    return bundle, next_starting_position


def positioning_bundle(bundle, starting_position):
    """
    positioning the bundle to start from starting_position, instead of [R|t] of the first pair. 
    for each pair, concat the trans from the first pair to the starting position.
    """
    R_news, t_news = [], []

    # find the new location of the first pair
    R, t = part2.get_Rt(starting_position)
    R_news.append(R.copy())
    t_news.append(t.copy())

    for i in range(1, len(bundle)):
        #finds the relative trans from the first pair to the current pair
        relative_trans = get_relative_transformation(bundle[0].extrinsic_left, bundle[i].extrinsic_left)
        R_relative, t_relative = part2.get_Rt(relative_trans)

        #concatenate the trans {world->starting position} to {first pair->current pair}
        R_next, t_next = concat_transformation(R_news[0], t_news[0], R_relative, t_relative)

        R_news.append(R_next.copy())
        t_news.append(t_next.copy())

    for i in range(len(bundle)):
        bundle[i].extrinsic_left = part2.hstack(R_news[i], t_news[i])


def concat_transformation(R1, t1, R2, t2):
    """
    concatenate transformations. given:
    p1 = R1 * p_world + t1
    p2 = R2 * p1 +t2
    --> P2 = R2 * R1 * p_world + R2 * t1 + t2 
    """
    return R2 @ R1, R2 @ t1 + t2

def get_inverted_transformation(R, t):
    return R.T, -R.T @ t


def extract_partial_path(pairs, tracks, start_idx, end_idx):
    new_pairs = []
    new_tracks = {}

    for pair in pairs[start_idx:end_idx+1]:
        new_pair = copy.deepcopy(pair)
        new_pairs.append(new_pair)

    #items = list(tracks.items())
    #for i in tqdm(range(len(items))):
    #    (index, track) = items[i]
    for (index, track) in tracks.items():
        if any(PairId in track.PairId_MatchIndex.keys() for PairId in range(pairs[start_idx].PairId, pairs[end_idx].PairId+1)):
            new_tracks[index] = track
        if index%1000 == 0:
            print(f'in index {index}\r', end="")
    
    return new_pairs, new_tracks


def get_relative_transformation(extrinsic1, extrinsic2):
    """
    get relative transformation from ext1 to ext2.
    :param extrinsic1: transrmation [R1|t1], s.t. p_camera1 = R1 @ p_world + t1
    :param extrinsic2: transrmation [R2|t2], s.t. p_camera2 = R2 @ p_world + t2
    :return: transformation [R|t],           s.t. p_camera2 = R @ p_camera1 + t
    """
    R1, t1 = part2.get_Rt(extrinsic1)
    R2, t2 = part2.get_Rt(extrinsic2)

    relative_R = R2 @ R1.T
    relative_t = np.add((-R2 @ R1.T @ t1), t2)
    return part2.hstack(relative_R, relative_t)
    

def get_average_distance(pairs, frames):
    """
    find the average of distance of {frames} frames
    """
    old_x, old_y = 0, 0
    sum, counter = 0, 0
    for i in range(10, 1000, frames):
        cv2.imshow(f'img', pairs[i].img_l)
        cv2.waitKey(100)
        p = part2.get_position_in_world(np.array([0, 0, 0]), *part2.get_Rt(pairs[i].extrinsic_left))
        new_x, new_y = p[0], p[2]
        dis = math.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
        sum += dis
        counter += 1
        print(f"dis from {i - 10} to {i} is {round(dis, 3)}")

        old_x, old_y = new_x, new_y

    avg = round(sum / counter, 3)
    print(f"the average distance is {avg}")
    return avg  # found 7.26 -> round = 7.3


if __name__ == "__main__":
    main()


"""
def convert_old_format_to_new_imgs(database_path):
    database = part2.Database(database_path)
    pairs = database.Pairs
    tracks = database.Tracks

    new_database = Database()
    new_pairs = new_database.Pairs
    new_tracks = new_database.Tracks

    for (i, p) in pairs.items():
        new_pair = Pair(p.PairId)  # creating pair, but without saving the images themselfs
        new_pair.matchIndex_TrackId = p.matchIndex_TrackId
        new_pair.extrinsic_left = p.extrinsic_left
        
        new_pairs[i] = new_pair

    for (i, t) in tracks.items():
            new_tracks[i] = t

    new_path = database_path[:-4] + ' - new_imgs.pkl'
    new_database.serialize(new_path)
    return new_database


def convert_old_format_to_new(database_path):
    
    # in the old format, Pairs had extrinsic_right, relative_extrinsic_left and cloud. the 'extrinsic_left' was absolute.
    # I found out that the three are redundant, and 'extrinsic_left' should be relative.
    # :param database_path: path of old database
    # :return: new_database
    
    database = part2.Database(database_path)
    pairs = database.Pairs
    tracks = database.Tracks

    new_database = Database()
    new_pairs = new_database.Pairs
    new_tracks = new_database.Tracks

    prev_transformation = part2.hstack(np.identity(3), np.zeros([3,]))

    for (i, p) in pairs.items():
        new_pair = Pair(p.PairId, p.img_l, p.img_r)
        new_pair.matchIndex_TrackId = p.matchIndex_TrackId

        # assuming [R|t] from world to camera, meaning: p_camera = R @ p_world + t
        #curr_R, curr_t = part2.get_Rt_from_extrinsic(p.extrinsic_left)
        #relative_R = curr_R @ prev_R.T
        #relative_t = np.add((-curr_R @ prev_R.T @ prev_t), curr_t)
        #new_pair.extrinsic_left = part2.hstack(relative_R, relative_t)
        new_pair.extrinsic_left = get_relative_transformation(prev_transformation, p.extrinsic_left)
        
        prev_transformation = p.extrinsic_left

        new_pairs[i] = new_pair

        if i%100 == 0:
            print(i)

    for (i, t) in tracks.items():
        new_tracks[i] = t

    new_path = database_path[:-4] + ' - new.pkl'
    new_database.serialize(new_path)
    return new_database


def show_camera_coords_relative(database=None):
    if database:
        locations = []
        R_sum, t_sum = np.identity(3), np.zeros([3,])
        for i in range(len(database.Pairs)):
            pair = database.Pairs[i]
            R_relative, t_relative = part2.get_Rt(pair.extrinsic_left)
            t_sum = np.add(R_relative @ t_sum, t_relative)
            R_sum = R_sum @ R_relative
            locations.append(part2.get_position_in_world(np.array([0, 0, 0]), R_sum, t_sum))

        # I'm not interested in y_axis, because it represents the height of the camera
        locations_0_2 = np.array([[l[0], l[2]] for l in locations])

        x, y = locations_0_2.T

        plt.scatter(x,y, c='red',s=np.array([5] * len(x)))  #

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
    plt.scatter(x, y, c='blue', s=np.array([5] * len(x)))  #

    plt.legend((['predicted', 'ground truth'] if database else ['ground truth']))

    plt.show()
"""