import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    #Compute gradients of current image
    h,w = It1.shape
    xaxis = np.arange(0,w)
    yaxis = np.arange(0,h)
    X,Y = np.meshgrid(xaxis,yaxis)
    interp1 = RectBivariateSpline(yaxis,xaxis,It1)
    gradX = interp1.ev(Y,X,dy=1)
    gradY = interp1.ev(Y,X,dx=1)
    dWdp = np.zeros((h,w,2,6))
    dWdp[:,:,0,0] = X
    dWdp[:,:,1,1] = X
    dWdp[:,:,0,2] = Y
    dWdp[:,:,1,3] = Y
    dWdp[:,:,0,4] = 1
    dWdp[:,:,1,5] = 1
    for i in range(15):
        #Warp image
        IW = scipy.ndimage.affine_transform(It1,M)
        #Compute error
        err = It-IW
        err[IW==0] = 0
        #Obtain gradI 
        gradXW = scipy.ndimage.affine_transform(gradX,M)
        gradYW = scipy.ndimage.affine_transform(gradY,M)
        gradI = np.stack([gradXW,gradYW],axis=-1)
        gradI = np.expand_dims(gradI,axis=2)
        #Compute Hessian
        gradI_J = np.matmul(gradI,dWdp)
        gradI_JT = np.transpose(gradI_J,axes=[0,1,3,2])
        G = np.matmul(gradI_JT,gradI_J)
        H = np.sum(G,axis=(0,1))
        """plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(It,cmap='gray')        
        plt.subplot(3, 2, 2)
        plt.imshow(IW,cmap='gray')        
        plt.subplot(3, 2, 3)
        plt.imshow(err,cmap='gray')        
        plt.subplot(3, 2, 4)
        plt.imshow(gradXW,cmap='gray')
        plt.subplot(3, 2, 5)
        plt.imshow(gradYW,cmap='gray')        
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()"""
        #Compute delta_p
        Hinv = np.linalg.inv(H)
        err = np.expand_dims(err,axis=2)
        err = np.expand_dims(err,axis=3)
        dp = np.matmul(gradI_JT,err)
        dpH = np.matmul(Hinv,dp)
        delta_p = np.sum(dpH,axis=(0,1)).reshape(6)
        p = p+delta_p
        M[0,0] = 1+p[3]
        M[1,0] = p[2]
        M[0,1] = p[1]
        M[1,1] = 1+p[0]
        M[0,2] = p[5]
        M[1,2] = p[4]
        if(np.linalg.norm(delta_p)<0.01):
            break
    return M
