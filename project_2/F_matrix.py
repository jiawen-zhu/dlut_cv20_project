import numpy as np
import pdb


# write your code here for part estimating fundamental matrix
def fit_fundamental(matches):
    """
    Solves for the fundamental matrix using the matches.
    """
    # matches: np array with size: (k, 4).
    # k is the match pairs number.
    # 4 is location in each match pair:[x_axis_img1, y_axis_img1, x_axis_img2, y_axis_img2]

    # return F: fundamental matrix, np array (3,3)

    M = np.zeros((matches.shape[0], 9))
    x1, x2, y1, y2 = matches[:, 0].T, matches[:, 2].T, matches[:, 1].T, matches[:, 3].T
    for i in range(matches.shape[0]):
        M[i] = [x1[i] * x2[i], x1[i] * y2[i], x1[i], y1[i] * x2[i], y1[i] * y2[i], y1[i], x2[i], y2[i], 1]
    U, S, V = np.linalg.svd(M)
    F_ori = V[-1].reshape(3, 3)
    U_, S_, V_ = np.linalg.svd(F_ori)
    S_[2] = 0
    A = np.dot(np.diag(S_), V_)
    F = np.dot(U_, A)

    # <YOUR CODE>

    return F


def plot_fundamental(ax, F, pt_2d, if_pt2):
    """
    function to  plot epipolar line function
    """

    one_ = np.array([[1.]])
    for i in range(len(pt_2d)):
        p_i = np.r_[pt_2d[i].reshape(2, 1), one_]
        if if_pt2:line_homogeneous = np.dot(F.T, p_i)
        else:line_homogeneous = np.dot(F, p_i)
        y = -line_homogeneous[2] / line_homogeneous[1]
        x = (-line_homogeneous[2]) / line_homogeneous[0]
        x1, y2 = 0, 0
        if x < 0:
            y2 = 712
            x = (-line_homogeneous[2] - line_homogeneous[1] * 712) / line_homogeneous[0]
        if y < 0:
            x1 = 1072
            y = (-line_homogeneous[2] - line_homogeneous[0] * 1072) / line_homogeneous[1]
        ax.plot([x1, x], [y, y2], 'g')


# <YOUR CODE>

    return ax
