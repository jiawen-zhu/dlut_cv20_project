import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from utils import *
from student_feature_matching import match_features
from student_sift import get_features
from student_harris import get_interest_points
from IPython.core.debugger import set_trace

#=========================================set up======================================
# Notre Dame
image1 = load_image('../data/Notre Dame/921919841_a30df938f2_o.jpg')
image2 = load_image('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')
eval_file = '../data/Notre Dame/Notre_Dame_match_ground_truth.pkl'

#gao_zilin@126.com

#for reduction the calculation consuming, we set a scale_factor to downsample the image.
#DO NOT CHANGE THIS FACTOR
scale_factor = 0.5
image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

feature_width = 16 # width and height of each local feature, in pixels. 


#================================harris corner detector==============================
x1, y1, R1_confidence = get_interest_points(image1_bw, feature_width)
x2, y2, R2_confidence = get_interest_points(image2_bw, feature_width)

# x1, y1, x2, y2, R1_confidence, R2_confidence = cheat_interest_points(
#                         eval_file, scale_factor)
#plt.figure(); plt.imshow(image1_bw)

# Visualize the interest points
c1 = show_interest_points(image1, x1, y1)
c2 = show_interest_points(image2, x2, y2)
plt.figure(); plt.imshow(c1); plt.show()
plt.figure(); plt.imshow(c2); plt.show()
print('{:d} corners in image 1, {:d} corners in image 2'.format(len(x1), len(x2)))


#================================SIFT feature encoding==============================
image1_features = get_features(image1_bw, x1, y1, feature_width)
image2_features = get_features(image2_bw, x2, y2, feature_width)

matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2)
print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

#================================visualization=================================
# num_pts_to_visualize = len(matches)
num_pts_to_visualize = 100
c1 = show_correspondence_circles(image1, image2,
                    x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                    x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
plt.figure(); plt.imshow(c1); plt.show()
plt.savefig('../results/vis_circles.jpg', dpi=1000)
c2 = show_correspondence_lines(image1, image2,
                    x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                    x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
plt.figure(); plt.imshow(c2); plt.show()
plt.savefig('../results/vis_lines.jpg', dpi=1000)


#================================evaluation=================================
evaluate_script(R1_confidence, R2_confidence,image1,image2,
    eval_file, scale_factor, x1, x2, y1, y2, matches,confidences)