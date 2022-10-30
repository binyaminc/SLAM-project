import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = r'D:\SLAM\exercises\VAN_ex\data\dataset05\sequences\05\\'

FILTER_RATIO = 0.75
NUM_MATCHES_SHOW = 20
DISTANCE_THREASHOLD = 400  # previously - 100

CYAN = (255,255,0)
ORANGE = (0, 69, 255)


def main():
    # ---------finding keyPoints with SIFT---------
    img1,img2 = read_images(0)  # img1 = queryImage, img2 = trainImage
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    garbage_output = 0
    # keypoints
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    img_1 = cv2.drawKeypoints(gray1,keypoints_1,garbage_output)
    img_2 = cv2.drawKeypoints(gray2, keypoints_2, garbage_output)

    cv2.imshow('res1',img_1)
    cv2.imshow('res2',img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------Matching the keyPoints---------

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < DISTANCE_THREASHOLD and m.distance < FILTER_RATIO * n.distance:
            good.append([m])  # append m as a list, so that we get list of lists as required in drawMatchesKnn
    good_before_height_filter = [i for i in good]

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good[:NUM_MATCHES_SHOW], garbage_output, flags=2)

    cv2.imshow('match',img3)

    # --------- plot histogram of deviations in height(#pixels) of matches---------
    show_hist_from_matches(good, keypoints_1, keypoints_2)

    # --------- filter matches if height differs in more than 2 pixels ---------
    good = list(filter(lambda m: abs(keypoints_1[m[0].queryIdx].pt[1] - keypoints_2[m[0].trainIdx].pt[1]) <= 2, good))
    print("amount of matches: ", len(good))
    show_hist_from_matches(good, keypoints_1, keypoints_2)

    # --------- present inlier and outlier keypoints ---------
    # take all the keypoints from 'matches'
    matched_kp1, matched_kp2 = get_matched_kps_from_matches(good_before_height_filter, keypoints_1, keypoints_2)
    img_1, img_2 = draw_matches(img1, img2, matched_kp1, matched_kp2, CYAN)

    # paint only the good ones --> only the outliers stay in cyan
    matched_kp1, matched_kp2 = get_matched_kps_from_matches(good, keypoints_1, keypoints_2)
    img_1, img_2 = draw_matches(img_1, img_2, matched_kp1, matched_kp2, ORANGE)

    cv2.imshow('matched_kp1', img_1)
    cv2.imshow('matched_kp2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------- Triangulation ----------
    k, m1, m2 = read_cameras()

    # matters of reshaping, hopefully
    np_kp1 = np.array([kp.pt for kp in matched_kp1])
    np_kp2 = np.array([kp.pt for kp in matched_kp2])

    # OpenCV implementation
    points_4d_hom = cv2.triangulatePoints(k @ m1, k @ m2, np_kp1.T, np_kp2.T)
    points_3d_hom = points_4d_hom[:3, :] / points_4d_hom[3, :]

    plot_3d_points(points_3d_hom)

    # my implementation
    my_points_4d_hom = my_generalized_triangulatePoints([k @ m1, k @ m2], [np_kp1, np_kp2])
    my_points_3d_hom = my_points_4d_hom[:3, :] / my_points_4d_hom[3, :]

    plot_3d_points(my_points_3d_hom)

    # ---------- present distance in X_axis of matched keypoints (for fun) ----------
    """
    print("distance in X_axis of matched keypoints")
    i = 0
    for kp1, kp2 in zip(matched_kp1, matched_kp2):
        print(str(i), kp1.pt[0] - kp2.pt[0])
        i += 1
    """
    input()


"""
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


def my_generalized_triangulatePoints(calib_matrices, kps):

    points_4d_hom = np.empty(shape=(4,0))

    # looping over all the points
    for i in range(len(kps[0])):

        # looping over the cameras and keypoint
        A = np.empty(shape=(0, 4))
        for j in range(len(calib_matrices)):
            m = calib_matrices[j]
            kp = kps[j][i]  # keypoint i of camera j
            equation_vector1 = m[2] * kp[0] - m[0]
            equation_vector2 = m[2] * kp[1] - m[1]

            A = np.vstack([A, equation_vector1, equation_vector2])

        # SVD on A, return the last column in V
        u,s,v_T = np.linalg.svd(A)
        points_4d_hom = np.c_[points_4d_hom, v_T[-1]]

    return points_4d_hom


def get_matched_kps_from_matches(matches, keypoints_1, keypoints_2):
    matched_indeces_kp1 = []
    matched_indeces_kp2 = []
    for match in matches:
        best_match = match[0]
        matched_indeces_kp1.append(best_match.queryIdx)
        matched_indeces_kp2.append(best_match.trainIdx)

    matched_kp1 = [keypoints_1[i] for i in matched_indeces_kp1]  # taking the actual keypoints and not only the indeces
    matched_kp2 = [keypoints_2[i] for i in matched_indeces_kp2]

    return matched_kp1, matched_kp2


def get_matched_kps_from_matches_with_matched_indeces(matches, keypoints_1, keypoints_2):
    matched_indeces_kp1 = []
    matched_indeces_kp2 = []
    for match in matches:
        best_match = match[0]
        matched_indeces_kp1.append(best_match.queryIdx)
        matched_indeces_kp2.append(best_match.trainIdx)

    matched_kp1 = [keypoints_1[i] for i in matched_indeces_kp1]  # taking the actual keypoints and not only the indeces
    matched_kp2 = [keypoints_2[i] for i in matched_indeces_kp2]

    matched_indeces_kp1 = dict(zip(matched_indeces_kp1, range(len(matched_indeces_kp1))))
    matched_indeces_kp2 = dict(zip(matched_indeces_kp2, range(len(matched_indeces_kp2))))

    return matched_kp1, matched_kp2, matched_indeces_kp1, matched_indeces_kp2


def draw_matches(base_img1, base_img2, matched_kp1, matched_kp2, my_color):

    garbage_output = 0
    img_1 = cv2.drawKeypoints(base_img1, matched_kp1, garbage_output, color=my_color)
    img_2 = cv2.drawKeypoints(base_img2, matched_kp2, garbage_output, color=my_color)

    return img_1, img_2


def plot_3d_points(points_3d, zlim=[], xlim=[], ylim=[]):
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
    plt.show()


def kp_to_npArray(kp):
    return np.array([list(v.pt) for v in kp])


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH+'image_0\\'+img_name)
    img2 = cv2.imread(DATA_PATH+'image_1\\'+img_name)
    return img1, img2


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:] # skip first token
        l2 = f.readline().split()[1:] # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2



def show_hist_from_matches(matches, keypoints_1, keypoints_2):
    dis = []
    sum_good = 0
    for match in matches:
        match = match[0]
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        temp_dis = abs(keypoints_1[idx1].pt[1] - keypoints_2[idx2].pt[1])
        dis.append(temp_dis)  # takes pt[1] because I compare with respect to Y axis
        if temp_dis <= 2:
            sum_good += 1

    #dis = [i for i in dis if i<50]
    plt.hist(dis, bins=100)
    plt.show()
    print(f"good percentage is {round(100*sum_good/len(matches), 2)} %")

    cv2.waitKey(0)


if __name__ == "__main__":
    main()


"""
given 2 images, 3 main steps are performed:
1. find (good) keypoints
2. matching the keypoints
3. get 3d points (triangulation)
"""

"""
# alternative way of matching, using 1to1 simple brute force and not Knn. But I think it's worse

    # create BFMatcher object
    bf = cv2.BFMatcher()#crossCheck=True)

    # Match descriptors.
    matches = bf.match(descriptors_1, descriptors_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = 0
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:100],img3, flags=2)

    cv2.imshow('match',img3)
    """