import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import SubtractDominantMotion as SDM

# write your script here, we recommend the above libraries for making your animation
frames = np.load("../data/aerialseq.npy")
_,_,T = frames.shape
fig,ax = plt.subplots(1)
im = ax.imshow(frames[:,:,0],cmap='gray')
plt.pause(0.01)
for i in range(1,T):
    moving = SDM.SubtractDominantMotion(frames[:,:,i-1],frames[:,:,i])
    im.set_data(frames[:,:,i])
    c = np.argwhere(moving!=0)
    if i==1:
        sc = ax.scatter(c[:,1],c[:,0],s=1,c='r')
    else:
        sc.remove()
        sc = ax.scatter(c[:,1],c[:,0],s=1,c='r')
    plt.draw()
    if(i==30):
        c1 = c.copy()
    if(i==60):
        c2 = c.copy()
    if(i==90):
        c3 = c.copy()
    if(i==120):
        c4 = c.copy()
    plt.pause(0.01)

#Save and plot results
plt.subplot(1,4,1)
plt.imshow(frames[:,:,30],cmap='gray')
ax1 = plt.gca()
ax1.scatter(c1[:,1],c1[:,0],s=1,c='r')
plt.subplot(1,4,2)
plt.imshow(frames[:,:,60],cmap='gray')
ax1 = plt.gca()
ax1.scatter(c2[:,1],c2[:,0],s=1,c='r')
plt.subplot(1,4,3)
plt.imshow(frames[:,:,90],cmap='gray')
ax1 = plt.gca()
ax1.scatter(c3[:,1],c3[:,0],s=1,c='r')
plt.subplot(1,4,4)
plt.imshow(frames[:,:,120],cmap='gray')
ax1 = plt.gca()
ax1.scatter(c4[:,1],c4[:,0],s=1,c='r')