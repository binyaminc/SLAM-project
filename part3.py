import math
import cv2
import numpy as np
import part1
import part2
from part2 import Database, Pair, Track, demi_match

AVG_DIS = 7.3  # average of distance passed in 10 frames


def main():
    database = Database('results\\data - 1 refine of PnP (1).pkl')
    pairs = database.Pairs
    tracks = database.Tracks

    avg_dis = get_average_distance(pairs, frames=10)  # found 7.26 -> round = 7.3
    print(f"average distance is {avg_dis}")


def get_average_distance(pairs, frames):
    """
    find the average of distance of {frames} frames
    """
    old_x, old_y = 0, 0
    sum, counter = 0, 0
    for i in range(10, 1000, frames):
        cv2.imshow(f'img', pairs[i].img_l)
        cv2.waitKey(100)
        p = part2.get_position_in_world(np.array([0, 0, 0]), *part2.extrinsic_to_Rt(pairs[i].extrinsic_left))
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
