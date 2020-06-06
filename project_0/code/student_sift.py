import numpy as np
import cv2
import pdb
import math


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). 

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    img_pad = np.pad(image, ((feature_width // 2, feature_width // 2), (feature_width // 2, feature_width // 2)), mode='constant')
    dx = cv2.Sobel(img_pad, cv2.CV_64F, 1, 0, ksize=15)
    dy = cv2.Sobel(img_pad, cv2.CV_64F, 0, 1, ksize=15)

    fs = []
    for k in range(len(x)):
        HG = np.zeros((4, 4, 8))
        for j in range(feature_width):
            for i in range(feature_width):
                bin = np.arctan2(dy[(int)(y[k]) + j][(int)(x[k]) + i], dx[(int)(y[k]) + j][(int)(x[k]) + i])
                mag = np.sqrt(dx[(int)(y[k]) + j][(int)(x[k]) + i] ** 2 + dy[(int)(y[k]) + j][(int)(x[k]) + i] ** 2)
                if bin > 1: bin = 2
                if bin < -1: bin = -1
                if dx[(int)(y[k]) + j][(int)(x[k]) + i] > 0: HG[(int)(j / 4)][(int)(i / 4)][math.ceil(bin + 1)] += mag
                else:HG[(int)(j / 4)][(int)(i / 4)][math.ceil(bin + 5)] += mag
        ft = np.reshape(HG, (1, 128))
        ft = ft / (ft.sum())
        fs.append(ft)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(fs)
