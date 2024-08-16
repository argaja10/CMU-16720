import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = np.zeros(2)
    #Get template image
    h,w = It.shape
    xaxis = np.arange(0,w)
    yaxis = np.arange(0,h)
    interp = RectBivariateSpline(yaxis,xaxis,It)
    x_s,y_s,x_e,y_e = rect
    n,m,_ = bases.shape
    Xr, Yr = np.meshgrid(np.linspace(x_s,x_e,m),np.linspace(y_s,y_e,n))
    T = interp.ev(Yr,Xr)
    #Create interpolation object for current image
    interp1 = RectBivariateSpline(yaxis,xaxis,It1)
    #Set threshold hyperparameter value
    threshold = 0.01
    B = bases.reshape(-1,bases.shape[-1])
    BBT = np.matmul(B,B.T)
    IBBT = np.eye(BBT.shape[0])-BBT
    while True:
        #Get current bounding rectangle on current image
        Xi = Xr+p[0]
        Yi = Yr+p[1]
        #Warp current image
        IW = interp1.ev(Yi,Xi)
        #Compute error
        if not np.array_equal(T.shape,IW.shape):
            break
        err = T-IW
        err = err.ravel()
        b = np.matmul(IBBT,err.reshape((-1,1))).ravel()
        #Warp gradient of current image
        IWx = interp1.ev(Yi,Xi,dy=1).ravel()
        IWy = interp1.ev(Yi,Xi,dx=1).ravel()
        A = np.stack([IWx,IWy],axis=-1)
        A = np.matmul(IBBT,A)
        Pinv = np.linalg.pinv(A)
        delta_p = np.matmul(Pinv,b)
        #Plot figures for diagnostics
        """fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(T)        
        plt.subplot(3, 2, 2)
        plt.imshow(IW)        
        plt.subplot(3, 2, 3)
        plt.imshow(err)        
        plt.subplot(3, 2, 4)
        plt.imshow(IWx)
        plt.subplot(3, 2, 5)
        plt.imshow(IWy)        
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()"""
        #Update parameter
        p = p + delta_p
        #Check if delta_p is smaller than threshold, break out of loop if true
        if np.linalg.norm(delta_p)<threshold:
            break
    return p
    
