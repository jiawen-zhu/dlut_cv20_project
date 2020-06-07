import numpy as np
import pdb

# Write your code here for camera calibration (lab)
def camera_calibration(pt_3d, matches):
    """
    write your code to compute camera matrix
    """
    #pt_3d: points location in 3d, np array with size: k x 3. [x,y,z].
    #k is the 3d points number.
    #3: 3 dimension: [x,y,z]

    #matches: np array with size: (k,4). 
    #k is the match pairs number. 
    #4 is location in each match pair:[x_axis_img1, y_axis_img1, x_axis_img2, y_axis_img2]

    #return P: project matrix: np array (3,4)
    #return K: fundamental matrix, np array (3,3)
    #return R: rotation matrix, np array (3,3)
    #return c: camera center, np array (4,)

    points_num = matches.shape[0]
    a = []
    b = []

    for n in range(points_num):
        x = pt_3d[n,0]
        y = pt_3d[n,1]
        z = pt_3d[n,2]
        u = matches[n,0]
        v = matches[n,1]
        a.append([x,y,z,1,0,0,0,0, -u*x, -u*y, -u*z])
        b.append([u])
        a.append([0,0,0,0,x,y,z,1, -v*x, -v*y, -v*z])
        b.append([v])

    A = np.mat(a)
    B = np.mat(b)
    M = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B))
    M = np.array(M.T)
    M = np.append(M,[1])
    P = np.reshape(M,(3,4))
    R, K = np.linalg.qr(P)
    c = np.dot(np.linalg.inv(np.dot(-P[:, 0:3].T, -P[:, 0:3])), np.dot(-P[:, 0:3].T, P[:, 3]))

    # <YOUR CODE>

    return P, K, R, c
