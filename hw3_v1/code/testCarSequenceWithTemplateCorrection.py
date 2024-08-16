import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK

# write your script here, we recommend the above libraries for making your animation
frames = np.load("../data/carseq.npy")
_,_,T = frames.shape
rect_init = np.asarray([59.,116.,145.,151.])
width = rect_init[2]-rect_init[0]
height = rect_init[3]-rect_init[1]
rect = rect_init.copy()
rects = np.zeros((T,4))
rects[0,:] = rect
fig,ax = plt.subplots(1)
im = ax.imshow(frames[:,:,0],cmap='gray')
rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rectangle)
plt.pause(0.01)
eps = 0.1
p0 = np.zeros(2)
for i in range(1,T):
    p = LK.LucasKanade(frames[:,:,i-1],frames[:,:,i],rect)
    rect[0] = rect[0] + p[0]
    rect[1] = rect[1] + p[1]
    rect[2] = rect[2] + p[0]
    rect[3] = rect[3] + p[1]
    p0 = p0+p
    p_star = LK.LucasKanade(frames[:,:,0],frames[:,:,i],rect_init,p0=p0)
    diff = np.linalg.norm(p0-p_star)
    if diff>eps:
        rect[0] = rect_init[0] + p_star[0]
        rect[1] = rect_init[1] + p_star[1]
        rect[2] = rect_init[2] + p_star[0]
        rect[3] = rect_init[3] + p_star[1]
        p0 = p_star
    rects[i,:] = rect
    im.set_data(frames[:,:,i])
    rectangle.remove()
    rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rectangle)    
    plt.draw()
    if(i==1):
        rect1 = rect.copy()
    if(i==100):
        rect2 = rect.copy()
    if(i==200):
        rect3 = rect.copy()
    if(i==300):
        rect4 = rect.copy()
    if(i==400):
        rect5 = rect.copy()
    plt.pause(0.01)
np.save('carseqrects-wcrt.npy',rects)
plt.subplot(1,5,1)
plt.imshow(frames[:,:,1],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect1[0],rect1[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,2)
plt.imshow(frames[:,:,100],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect2[0],rect2[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,3)
plt.imshow(frames[:,:,200],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect3[0],rect3[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,4)
plt.imshow(frames[:,:,300],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect4[0],rect4[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,5)
plt.imshow(frames[:,:,400],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect5[0],rect5[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)