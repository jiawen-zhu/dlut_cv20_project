import numpy as np
import pdb


# write your code here for part estimating fundamental matrix
def fit_fundamental(matches):
    """
    Solves for the fundamental matrix using the matches.
    """
    #matches: np array with size: (k, 4). 
    #k is the match pairs number. 
    #4 is location in each match pair:[x_axis_img1, y_axis_img1, x_axis_img2, y_axis_img2]

    #return F: fundamental matrix, np array (3,3)

    n = matches.shape[0]
    x1 = matches[:, 0].T
    x2 = matches[:, 2].T
    y1 = matches[:, 1].T
    y2 = matches[:, 3].T
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[i] * x2[i], x1[i] * y2[i], x1[i], y1[i] * x2[i], y1[i] * y2[i], y1[i], x2[i], y2[i], 1]
    # 计算线性最小二乘解
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # 受限 F
    # 通过将最后一个奇异值置 0，使秩为 2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    # <YOUR CODE>

    return F


def plot_fundamental(ax, F, pt_2d, if_pt2):
    """
    function to  plot epipolar line function
    """
    one_ = np.array([[1.]])

    if not if_pt2:
        for i in range(len(pt_2d)):
            p_i = pt_2d[i].reshape(2, 1)
            p_i = np.r_[p_i, one_]
            line_homogeneous = np.dot(F, p_i)
            y = -line_homogeneous[2] / line_homogeneous[1]
            x = (-line_homogeneous[2] - line_homogeneous[1] * 712) / line_homogeneous[0]
            if x < 0:
                x = (-line_homogeneous[2]) / line_homogeneous[0]
                y2 = 0
            else:
                y2 = 712
            if y < 0:
                y = (-line_homogeneous[2] - line_homogeneous[0]*1072) / line_homogeneous[1]
                x1 = 1072
            else:
                x1 = 0
            ax.plot([x1, x], [y, y2], 'r')
    else:
        for i in range(len(pt_2d)):
            p_i = pt_2d[i].reshape(2, 1)
            p_i = np.r_[p_i, one_]
            line_homogeneous = np.dot(F.T, p_i)
            y = (-line_homogeneous[2]) / line_homogeneous[1]
            x = (-line_homogeneous[2]- line_homogeneous[1] * 712) / line_homogeneous[0]
            if x < 0:
                x = (-line_homogeneous[2]) / line_homogeneous[0]
                y2 = 0
            else:
                y2 = 712
            if y < 0:
                y = (-line_homogeneous[2] - line_homogeneous[0] * 1072) / line_homogeneous[1]
                x1 = 1072
            else:
                x1 = 0
            ax.plot([x1, x], [y, y2], 'r')

    # <YOUR CODE>


    return ax