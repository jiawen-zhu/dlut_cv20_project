import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


def get_interest_points(img, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   R_sort: A numpy array of shape (N,), descend sorted R,  correspondences to
                x, y: R_sort[i] =   R[y[i],x[i]]

    please notice that cv2 returns the image with [h,w], which correspondes [y,x] dim respectively. 
    ([vertically direction, horizontal direction])
    """
 
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    filtered_ = cv2.filter2D(img, -1, cv2.getGaussianKernel(3, 2))
    dx = cv2.Sobel(filtered_, cv2.CV_64F, 1, 0, ksize=15)
    dy = cv2.Sobel(filtered_, cv2.CV_64F, 0, 1, ksize=15)
    Ixy = dx * dy
    Ix2 = dx ** 2
    Iy2 = dy ** 2

    dxy_w = cv2.filter2D(Ixy, -1, cv2.getGaussianKernel(9, 2))
    dx2_w = cv2.filter2D(Ix2, -1, cv2.getGaussianKernel(9, 2))
    dy2_w = cv2.filter2D(Iy2, -1, cv2.getGaussianKernel(9, 2))

    alp = 0.06
    R = dx2_w * dy2_w - dxy_w ** 2 - alp * (dx2_w + dy2_w) ** 2

    corner_points = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            corner_points.append([R[i, j], j, i])
    sorted_points = sorted(corner_points, key=lambda x: x[0], reverse=True)
    sorted_points = sorted_points[0:10000]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    #############################################################################

    radi = []
    N = len(sorted_points)
    for i in range(N):
        r_sqaure = 1e10
        point1 = sorted_points[i]
        for j in range(i):
            point2 = sorted_points[j]
            temp = (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2
            r_sqaure = min(temp, r_sqaure)
        radi.append([np.sqrt(r_sqaure), point1[0], point1[1], point1[2]])

    points_ = sorted(radi, key=lambda x: x[0], reverse=True)
    x = np.array([])
    y = np.array([])
    for item in points_[:1500]:
        x=np.append(x, [item[2]], axis=0)
    for item in points_[:1500]:
        y=np.append(y, [item[3]], axis=0)
    R_sort=None

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y,R_sort


