from tqdm import tqdm
import copy
import gtsam
from gtsam.utils import plot
import pickle
import numpy as np
from matplotlib import pyplot as plt

import part2
from part2 import Database, Pair, Track, demi_match
import part3
from part3 import get_seperating_indeces, solve_bundle, positioning_bundle, POSE_UNCERTAINTY

SOURCE_DATA = r'results/data 05.pkl'


def main():
    # loading the data from part2
    database_path = SOURCE_DATA 
    database = Database(database_path)

    # get the indeces that seperate the images to bundles
    seperating_idxes = get_seperating_indeces([p.extrinsic_left for p in database.Pairs])  # the indexes that seperate bundles
    
    # for each bundle, get the (global) poses of the keyframes, and their (relative) covariances
    poses, covariances = get_poses_and_covariances(database.Pairs, seperating_idxes)
    extrinsics_old = get_extrinsics_from_poses(poses)

    # solve pose graph
    poses, result, graph = solve_pose_graph(poses, covariances, seperating_idxes)
    extrinsics_new = get_extrinsics_from_poses(poses)

    # plot locations of keyframes, old and new
    plot_locations(extrinsics_old, extrinsics_new)

    # plot locations with their covariances
    plot.plot_trajectory(1, result, marginals=gtsam.Marginals(graph, result), scale=1, title='Poses with covariances')
    plot.set_axes_equal(1)
    plt.show()


def plot_locations(exts1, exts2):
    # add the old locations to plot
    locations_0_2 = part2.get_extrinsics_locations(exts1)
    x, y = locations_0_2.T
    plt.scatter(x,y, c='red',s=np.array([5] * len(x)))

    # add the new locations to plot
    locations_0_2 = part2.get_extrinsics_locations(exts2)
    x, y = locations_0_2.T
    plt.scatter(x, y, c='blue', s=np.array([5] * len(x)))

    plt.legend(['old location', 'new location'])
    plt.show()
    

def solve_pose_graph(poses, covariances, seperating_idxes):

    # Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # give prior knowledge on the beginning position of the pose graph
    keyframe0 = gtsam.symbol('c', 0)
    graph.add(gtsam.PriorFactorPose3(keyframe0, poses[0], POSE_UNCERTAINTY))

    # create factors between two adjacent keyframes
    for i in range(len(poses) - 1):
        # calculate the relative pose
        relative_pose = poses[i].between(poses[i+1])
        
        # get symbols of adjacent keyframes
        prev_keyframe = gtsam.symbol('c', seperating_idxes[i])
        next_keyframe = gtsam.symbol('c', seperating_idxes[i + 1])
        
        between_factor = gtsam.BetweenFactorPose3(prev_keyframe, next_keyframe, relative_pose, covariances[i+1])
        graph.add(between_factor)

    # fill initial estimate
    for i in range(len(poses)):
        curr_keyframe = gtsam.symbol('c', seperating_idxes[i])
        initial_estimate.insert(curr_keyframe, poses[i])

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    print(f"error: {optimizer.error()}")
    result = optimizer.optimize()
    print(f'Final Error: {optimizer.error()}')

    # update poses
    for i in range(len(poses)):
        poses[i] = result.atPose3(gtsam.symbol('c', seperating_idxes[i]))
        
    return poses, result, graph


def get_extrinsics_from_poses(poses):
    extrinsics = []

    for i in range(len(poses)):
        pose = poses[i]
        R = pose.rotation().matrix()
        t = np.array([pose.x(), pose.y(), pose.z()])
        R, t = part3.get_inverted_transformation(R, t)
        extrinsics.append(part2.hstack(R, t))
    
    return extrinsics 


def get_poses_and_covariances(pairs, seperating_idxes):
    poses, covariances = [], []
    first_bundle = True
    starting_position = pairs[0].extrinsic_left.copy()  # starting position of new bundle
    
    for i in tqdm(range(len(seperating_idxes) - 1)):
    
        # start new bundle
        bundle = copy.deepcopy(pairs[seperating_idxes[i]: seperating_idxes[i+1] + 1])
        print(f'\rbundle from {seperating_idxes[i]} to {seperating_idxes[i+1]}  ', end="")
        
        adjed_bundle, result, graph = solve_bundle(bundle)

        # positioning the bundle to begin from the starting position - end of the last bundle.
        positioning_bundle(adjed_bundle, starting_position)
        starting_position = adjed_bundle[-1].extrinsic_left.copy()
        
        start_keyframe = gtsam.symbol('c', seperating_idxes[i])  # I'm using global numbering, by gut fealing it will be good
        end_keyframe = gtsam.symbol('c', seperating_idxes[i+1])

        if first_bundle:
            poses.append(result.atPose3(start_keyframe))  # taking the location, in world to camera trans, as gtsam expects. 
            covariances.append(part3.POSE_UNCERTAINTY)
            first_bundle = False
        
        # add initial for the last pair
        poses.append(result.atPose3(end_keyframe))
        
        # get covariance for the end keyframe
        marginals = gtsam.Marginals(graph, result)
        keys = gtsam.KeyVector()
        keys.append(start_keyframe)  # the first camera index in the bundle
        keys.append(end_keyframe)  # the index of the last camera in bundle
        marginal_covariance = marginals.jointMarginalCovariance(keys).fullMatrix()
        information_mat = np.linalg.inv(marginal_covariance)
        covariances.append(gtsam.noiseModel.Gaussian.Covariance(np.linalg.inv(information_mat[6:, 6:])))
        
    return poses, covariances

if __name__ == '__main__':
    main()

