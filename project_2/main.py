from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import F_matrix 
import camera_calibration
import util 

# Fundamental matrix estimation
name = 'lab'

I1 = Image.open('./data/{:s}1.jpg'.format(name))
I2 = Image.open('./data/{:s}2.jpg'.format(name))
I1_data = np.array(I1)
I2_data = np.array(I2)
matches = np.loadtxt('./data/{:s}_matches.txt'.format(name))
#matches: np array with size: k x 4. 
#k is the match pairs number. 
#4 is location in each match pair:[x_axis_img1, y_axis_img1, x_axis_img2, y_axis_img2]

N = len(matches) # 20

#--------------------Visualization---------------------
## Display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
I3 = np.zeros((I1.size[1],I1.size[0]*2,3))
I3[:,:I1.size[0],:] = I1
I3[:,I1.size[0]:,:] = I2
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.plot(matches[:,0],matches[:,1],  '+r')
# ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
# ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
# ax.plot([100,300],[200,500], 'r')
# ax.imshow(np.array(I3).astype(np.uint8))
# plt.show()

#--------------------Fundamental Matrix---------------------
# non-normalized method
F = F_matrix.fit_fundamental(matches) # <YOUR CODE>
pt1_2d = matches[:, :2] #points location in lab1
pt2_2d = matches[:, 2:] #points location in lab2
# Plot epipolar lines in image I2
fig, ax = plt.subplots()
# ax.set_aspect('equal')
ax = F_matrix.plot_fundamental(ax, F, pt1_2d, True) # <YOUR CODE>
ax.plot(matches[:,2],matches[:,3],  '+r')
ax.imshow(np.array(I2).astype(np.uint8))
plt.show()

# Plot epipolar lines in image I1
fig, ax = plt.subplots()
ax = F_matrix.plot_fundamental(ax, F, pt2_2d, False) # <YOUR CODE>
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.imshow(np.array(I1).astype(np.uint8))
plt.show()
print("fundamental matrix")
print("{}\n".format(F))



#--------------------camera.  calibration----------------------
# Load 3D points, and their corresponding locations in 
# the two images.
pts_3d = np.loadtxt('./data/lab_3d.txt')
matches = np.loadtxt('./data/lab_matches.txt')
pt1_2d = matches[:, :2]
pt2_2d = matches[:, 2:]

# <YOUR CODE> print lab camera projection matrices:
lab1_proj, lab1_K, lab1_R, lab1_c = \
              camera_calibration.camera_calibration(pts_3d, pt1_2d)# <YOUR CODE>
lab2_proj, lab2_K, lab2_R, lab2_c = \
              camera_calibration.camera_calibration(pts_3d, pt2_2d)# <YOUR CODE>


print('lab 1 camera projection P \n {} \n'.format(lab1_proj))
print('lab 1 intrinsic matrix K \n {} \n' .format(lab1_K))
print('lab 1 rotation matrix R \n {} \n'.format(lab1_R))
print('lab 1 camera center c \n {} \n'.format(lab1_c))


print('lab 2 camera projection P \n {} \n'.format(lab2_proj))
print('lab 2 intrinsic matrix K \n {} \n' .format(lab2_K))
print('lab 2 rotation matrix R \n {} \n'.format(lab2_R))
print('lab 2 camera center c \n {} \n'.format(lab2_c))

# evaluate the residuals for both estimated cameras
_, lab1_res = util.evaluate_points(lab1_proj, matches[:,:2], pts_3d)
print('residuals between the observed 2D points and the projected 3D points:')
print('residual in lab1:', lab1_res)
_, lab2_res = util.evaluate_points(lab2_proj, matches[:,2:], pts_3d)
print('residual in lab2:', lab2_res)