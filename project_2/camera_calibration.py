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

    
    # <YOUR CODE>

    return P, K, R, c
