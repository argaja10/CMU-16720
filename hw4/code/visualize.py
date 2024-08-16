'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import submission as sub
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import helper

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
exec(open("findM2.py").read())
query_D = np.load("../data/templeCoords.npz")
x1 = query_D['x1']
y1 = query_D['y1']
pts1 = np.concatenate((x1,y1),axis=1)
N = x1.shape[0]
pts2 = np.zeros((N,2),dtype=int)
for i in range(N):
    pts2[i,0], pts2[i,1] = sub.epipolarCorrespondence(im1,im2,F,x1[i],y1[i]) 

P,_ = sub.triangulate(C1,pts1,C2,pts2)
np.savez('q4_2.npz',F=F,M1=M1,M2=M2,C1=C1,C2=C2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], c='b', marker='o')
plt.show()

