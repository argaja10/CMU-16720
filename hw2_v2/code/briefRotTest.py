# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import cv2
import BRIEF as brf

img = cv2.imread('../data/model_chickenbroth.jpg')
compareX, compareY = np.load('../results/testPattern.npy')
h,w,d = img.shape
angle_list = np.arange(0,360+1,10)
locs,desc = brf.briefLite(img)
num_matches = []
for angle in angle_list:
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(w,h))
    locs_rot,desc_rot = brf.briefLite(img_rot)
    matches = brf.briefMatch(desc, desc_rot)
    #brf.plotMatches(img,img_rot,matches,locs,locs_rot)
    num_matches.append(matches.shape[0])
num_matches = np.asarray(num_matches)
plt.figure()
plt.bar(angle_list,num_matches)
plt.xlabel('Angle of rotation')
plt.ylabel('No. of matches')
plt.savefig('../results/q2_5.jpg')
plt.show()