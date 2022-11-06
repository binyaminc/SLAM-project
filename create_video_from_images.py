import part2
from matplotlib import pyplot as plt
import cv2
import numpy as np

DIR_PATH = r'D:\SLAM\exercises\VAN_ex\data\dataset05\sequences\05\image_0'
IMGS_COUNT = 2760

def main():
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.mp4', fourcc, 10, (1226, 370))

    for j in range(0, IMGS_COUNT+1):
        img_path = DIR_PATH + f"\\{j:06}" + ".png"
        img = cv2.imread(img_path)
        video.write(img)
        print(f"img {j} in")

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
