import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


DATA_PATH = 'data/non-rectified/'
SAVE_DIR = 'data/epipolar_line/'
random.seed(996)


def read_imgs():
    cam1_list = []
    cam2_list = []

    for i in range(10):
        img1_name = DATA_PATH + '/0' + str(i) + '/im0.png'
        img2_name = DATA_PATH + '/0' + str(i) + '/im1.png'
        img0 = cv2.imread(img1_name, 0)  # queryimage # left image. 0 is loading grey image
        img1 = cv2.imread(img2_name, 0)  # trainimage # right image
        cam1_list.append(img0)
        cam2_list.append(img1)

    return cam1_list, cam2_list


def show_imgs(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def find_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return pts1, pts2


def compute_fundamental(x1, x2):
    """
    Computes the fundamental matrix from corresponding points (x1,x2 3*n arrays) using the 8 point algorithm.
    Each row in the A matrix below is constructed as [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    # check if x1 and x2 has the same dimension
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F / F[2, 2]


def F_from_ransac(x1, x2, model, maxiter=5000, match_theshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).
        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    import ransac

    data = np.vstack((x1, x2))

    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_theshold, 20, return_all=True)
    return F, ransac_data['inliers']


def compute_fundamental_matrix(pts1, pts2):
    # show keypoint
    pts1 = np.array(pts1, dtype=np.int32)
    pts2 = np.array(pts2, dtype=np.int32)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    F = compute_fundamental(pts1, pts2)

    # We select only inlier points
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    """
    img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # randomly choose 5 points from line, pts1 adn pts2
    rand_point = []
    for i in range(5):
        rand_value = random.randint(0, len(pts1))
        rand_point.append(rand_value)

    for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        if i in rand_point:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, thickness=2, lineType=cv2.LINE_AA)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def draw_epi(img1, img2):
    pts1, pts2 = find_matches(img1, img2)
    F, pts1, pts2 = compute_fundamental_matrix(pts1, pts2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img_epi_1, img_pts_1 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img_epi_2, img_pts_2 = drawlines(img2, img1, lines2, pts2, pts1)

    return img_epi_1, img_pts_1, img_epi_2, img_pts_2


if __name__ == "__main__":

    cam1_list, cam2_list = read_imgs()

    for i in range(10):
        # compute epipolar line and impose on picture
        img_epi_1, img_pts_1, img_epi_2, img_pts_2 = draw_epi(cam1_list[i], cam2_list[i])

        # save graph
        plt.title("the epiploar line for image pair" + str(i))
        plt.subplot(221), plt.imshow(img_pts_1)
        plt.subplot(222), plt.imshow(img_epi_1)
        plt.subplot(223), plt.imshow(img_pts_2)
        plt.subplot(224), plt.imshow(img_epi_2)
        plt.savefig(fname=SAVE_DIR+"epipolar_"+str(i)+".png")

        # print
        print("image {} has been saved".format(i))
