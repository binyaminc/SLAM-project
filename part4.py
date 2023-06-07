from tqdm import tqdm
import copy
import gtsam
import pickle
import numpy as np

from part2 import Database, Pair, Track, demi_match
import part3
from part3 import get_seperating_indeces, solve_bundle, positioning_bundle

SOURCE_DATA = r'results/data 05.pkl'


def main():
    # loading the data from part2
    database_path = SOURCE_DATA 
    database = Database(database_path)

    initials, covariances = get_initials_and_covariances(database.Pairs)

    x=0
    
    

def get_initials_and_covariances(pairs):
    initials, covariances = [], []
    first_bundle = True
    starting_position = pairs[0].extrinsic_left.copy()  # starting position of new bundle
    
    # seperating_idxes = get_seperating_indeces([p.extrinsic_left for p in pairs])  # the indexes that seperate bundles. Each is the end of previous and start of new bundle
    seperating_idxes = []
    with open('seperating_indeces.pkl', 'rb') as f:
            seperating_idxes = pickle.load(f)
    seperating_idxes = [0] + (seperating_idxes)

    start_idx = seperating_idxes[0]

    tries = 0

    for i in tqdm(seperating_idxes[1:]):
    
        # start new bundle
        end_idx = i

        # start_idx = 138
        # end_idx = 147
        
        bundle = copy.deepcopy(pairs[start_idx: end_idx + 1])
        print(f'\rbundle from {start_idx} to {end_idx}  ', end="")
        
        positioning_bundle(bundle, starting_position)
        
        adjed_bundle, result, graph = solve_bundle(bundle)
        
        # positioning the bundle to start from the starting position - end of the last bundle.
        positioning_bundle(adjed_bundle, starting_position)
        starting_position = adjed_bundle[-1].extrinsic_left.copy()
        
        if first_bundle:
            initials.append(adjed_bundle[0].extrinsic_left)
            covariances.append(part3.POSE_UNCERTAINTY)
            first_bundle = False
        
        # get covariance for the last pair (end keyframe)
        try:
            marginals = gtsam.Marginals(graph, result)
            keys = gtsam.KeyVector()
            keys.append(gtsam.symbol('c', 0))  # the first camera index in the bundle
            keys.append(gtsam.symbol('c', end_idx - start_idx))  # the index of the last camera in bundle
            marginal_covariance = marginals.jointMarginalCovariance(keys).fullMatrix()
            information_mat = np.linalg.inv(marginal_covariance)
            covariances.append(np.linalg.inv(information_mat[6:, 6:]))
            print(f"success {tries}")
        except Exception as e:
            print(f"problem {tries}")
        
        tries += 1
        
        # add initial for the last pair
        initials.append(adjed_bundle[-1].extrinsic_left)
        start_idx = end_idx  # start of the next bundle is the end of the previous one

    return initials, covariances

if __name__ == '__main__':
    main()

