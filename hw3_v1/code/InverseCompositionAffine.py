import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage

def InverseCompositionAffine(It, It1):
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
    interp = RectBivariateSpline(yaxis,xaxis,It)
    gradXT = interp.ev(Y,X,dy=1)
    gradYT = interp.ev(Y,X,dx=1)
    gradT = np.stack([gradXT,gradYT],axis=-1)
    gradT = np.expand_dims(gradT,axis=2)
    dWdp = np.zeros((h,w,2,6))
    dWdp[:,:,0,0] = X
    dWdp[:,:,1,1] = X
    dWdp[:,:,0,2] = Y
    dWdp[:,:,1,3] = Y
    dWdp[:,:,0,4] = 1
    dWdp[:,:,1,5] = 1
    gradT_J = np.matmul(gradT,dWdp)
    gradT_JT = np.transpose(gradT_J,axes=[0,1,3,2])
    G = np.matmul(gradT_JT,gradT_J)
    H = np.sum(G,axis=(0,1))
    Hinv = np.linalg.inv(H)
    for i in range(15):
        #Warp image
        IW = scipy.ndimage.affine_transform(It1,M)
        #Compute error
        err = IW-It
        err[IW==0] = 0
        #Obtain gradI 
        err = np.expand_dims(err,axis=2)
        err = np.expand_dims(err,axis=3)
        dp = np.matmul(gradT_JT,err)
        dpH = np.matmul(Hinv,dp)
        delta_p = np.sum(dpH,axis=(0,1)).reshape(6)
        delta_p = inverseAffine(delta_p)
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

def inverseAffine(delta_p):
    dp = np.zeros(6)
    det = (1+delta_p[0])*(1+delta_p[3]) - delta_p[1]*delta_p[2]
    dp[0] = delta_p[1]*delta_p[2] - delta_p[0]*delta_p[3] - delta_p[0]
    dp[1] = -delta_p[1]
    dp[2] = -delta_p[2]
    dp[3] = delta_p[1]*delta_p[2] - delta_p[0]*delta_p[3] - delta_p[3]
    dp[4] = delta_p[2]*delta_p[5] - delta_p[3]*delta_p[4] - delta_p[4]
    dp[5] = delta_p[1]*delta_p[4] - delta_p[0]*delta_p[5] - delta_p[5]
    return dp/det