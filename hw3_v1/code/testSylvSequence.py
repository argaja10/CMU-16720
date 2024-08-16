import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis as LKB
import LucasKanade as LK

# write your script here, we recommend the above libraries for making your animation
frames = np.load("../data/sylvseq.npy")
bases = np.load("../data/sylvbases.npy")
_,_,T = frames.shape
rect_init = np.asarray([101.,61.,155.,107.])
width = rect_init[2]-rect_init[0]
height = rect_init[3]-rect_init[1]
rect_b = rect_init.copy()
rect = rect_init.copy()
rects = np.zeros((T,4))
rects[0,:] = rect_b
fig,ax = plt.subplots(1)
im = ax.imshow(frames[:,:,0],cmap='gray')
rectangle_b = patches.Rectangle((rect_b[0],rect_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax.add_patch(rectangle_b)
rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax.add_patch(rectangle)
plt.pause(0.01)
for i in range(1,T):
    p_b = LKB.LucasKanadeBasis(frames[:,:,i-1],frames[:,:,i],rect_b,bases)
    rect_b[0] = rect_b[0] + p_b[0]
    rect_b[1] = rect_b[1] + p_b[1]
    rect_b[2] = rect_b[2] + p_b[0]
    rect_b[3] = rect_b[3] + p_b[1]
    rects[i,:] = rect_b
    p = LK.LucasKanade(frames[:,:,i-1],frames[:,:,i],rect)
    rect[0] = rect[0] + p[0]
    rect[1] = rect[1] + p[1]
    rect[2] = rect[2] + p[0]
    rect[3] = rect[3] + p[1]
    im.set_data(frames[:,:,i])
    rectangle_b.remove()
    rectangle.remove()
    rectangle_b = patches.Rectangle((rect_b[0],rect_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
    ax.add_patch(rectangle_b)
    rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rectangle)    
    plt.draw()
    if(i==1):
        rect1_b = rect_b.copy()
        rect1 = rect.copy()
    if(i==200):
        rect2_b = rect_b.copy()
        rect2 = rect.copy()
    if(i==300):
        rect3_b = rect_b.copy()
        rect3 = rect.copy()
    if(i==350):
        rect4_b = rect_b.copy()
        rect4 = rect.copy()
    if(i==400):
        rect5_b = rect_b.copy()
        rect5 = rect.copy()
    plt.pause(0.01)
    
#Save and plot results
np.save('sylvrects.npy',rects)
plt.subplot(1,5,1)
plt.imshow(frames[:,:,1],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect1[0],rect1[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
rectangle = patches.Rectangle((rect1_b[0],rect1_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,2)
plt.imshow(frames[:,:,200],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect2[0],rect2[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
rectangle = patches.Rectangle((rect2_b[0],rect2_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,3)
plt.imshow(frames[:,:,300],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect3[0],rect3[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
rectangle = patches.Rectangle((rect3_b[0],rect3_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,4)
plt.imshow(frames[:,:,350],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect4[0],rect4[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
rectangle = patches.Rectangle((rect4_b[0],rect4_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax1.add_patch(rectangle)
plt.subplot(1,5,5)
plt.imshow(frames[:,:,400],cmap='gray')
ax1 = plt.gca()
rectangle = patches.Rectangle((rect5[0],rect5[1]),width,height,linewidth=1,edgecolor='b',facecolor='none')
ax1.add_patch(rectangle)
rectangle = patches.Rectangle((rect5_b[0],rect5_b[1]),width,height,linewidth=1,edgecolor='y',facecolor='none')
ax1.add_patch(rectangle)