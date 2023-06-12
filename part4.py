from tqdm import tqdm
import copy
import gtsam
import pickle
import numpy as np

from part2 import Database, Pair, Track, demi_match
import part3
from part3 import get_seperating_indeces, solve_bundle, positioning_bundle, POSE_UNCERTAINTY

SOURCE_DATA = r'results/data 05.pkl'


def main():
    # loading the data from part2
    database_path = SOURCE_DATA 
    database = Database(database_path)

    seperating_idxes = get_seperating_indeces([p.extrinsic_left for p in database.Pairs])  # the indexes that seperate bundles
    seperating_idxes = [0] + seperating_idxes

    poses, covariances = get_poses_and_covariances(database.Pairs, seperating_idxes)

    # Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # give prior knowledge on the beginning position of the pose graph
    keyframe0 = gtsam.symbol('c', 0)
    graph.add(gtsam.PriorFactorPose3(keyframe0, poses[0], POSE_UNCERTAINTY))

    # create factors between two adjacent keyframes
    for i in range(len(seperating_idxes) - 1):
        # calculate the relative pose
        relative_pose = poses[i].between(poses[i+1])
        
        # get symbols of adjacent keyframes
        prev_keyframe = gtsam.symbol('c', seperating_idxes[i])
        next_keyframe = gtsam.symbol('c', seperating_idxes[i + 1])
        
        between_factor = gtsam.BetweenFactorPose3(prev_keyframe, next_keyframe, relative_pose, covariances[i+1])
        graph.add(between_factor)

    # fill initial estimate
    for i in range(len(seperating_idxes)):
        curr_keyframe = gtsam.symbol('c', seperating_idxes[i])
        initial_estimate.insert(curr_keyframe, poses[i])



def get_poses_and_covariances(pairs, seperating_idxes):
    poses, covariances = [], []
    first_bundle = True
    starting_position = pairs[0].extrinsic_left.copy()  # starting position of new bundle
    
    start_idx = seperating_idxes[0]

    for i in tqdm(seperating_idxes[1:]):
    
        # start new bundle
        end_idx = i

        bundle = copy.deepcopy(pairs[start_idx: end_idx + 1])
        print(f'\rbundle from {start_idx} to {end_idx}  ', end="")
        
        # TODO: take out computation of new locations from 'solve_bundle'
        adjed_bundle, result, graph = solve_bundle(bundle)
        
        # # positioning the bundle to start from the starting position - end of the last bundle.
        # positioning_bundle(adjed_bundle, starting_position)
        # starting_position = adjed_bundle[-1].extrinsic_left.copy()
        
        start_keyframe = gtsam.symbol('c', start_idx)  # I'm using global numbering, by gut fealing it will be good
        end_keyframe = gtsam.symbol('c', end_idx)

        if first_bundle:
            poses.append(result.atPose3(start_keyframe))  # taking the location, in world to camera trans, as gtsam expects. 
            covariances.append(part3.POSE_UNCERTAINTY)
            first_bundle = False
        
        # add initial for the last pair
        #initials.append(adjed_bundle[-1].extrinsic_left)
        poses.append(result.atPose3(end_keyframe))
        
        # get covariance for the end keyframe
        marginals = gtsam.Marginals(graph, result)
        keys = gtsam.KeyVector()
        keys.append(start_keyframe)  # the first camera index in the bundle
        keys.append(end_keyframe)  # the index of the last camera in bundle
        marginal_covariance = marginals.jointMarginalCovariance(keys).fullMatrix()
        information_mat = np.linalg.inv(marginal_covariance)
        covariances.append(np.linalg.inv(information_mat[6:, 6:]))
        
        start_idx = end_idx  # start of the next bundle is the end of the previous one

    return poses, covariances

if __name__ == '__main__':
    main()

