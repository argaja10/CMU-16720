import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    p = p0
    #Get template image
    h,w = It.shape
    xaxis = np.arange(0,w)
    yaxis = np.arange(0,h)
    interp = RectBivariateSpline(yaxis,xaxis,It)
    x_s,y_s,x_e,y_e = rect
    Xr, Yr = np.mgrid[x_s:x_e+1,y_s:y_e+1]
    T = interp.ev(Yr,Xr)
    #Create interpolation object for current image
    interp1 = RectBivariateSpline(yaxis,xaxis,It1)
    #Set threshold hyperparameter value
    threshold = 0.01
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
        #Warp gradient of current image
        IWx = interp1.ev(Yi,Xi,dy=1)
        IWy = interp1.ev(Yi,Xi,dx=1)
        #No need to compute Jacobian in this example, it is identity matrix
        #Compute Hessian
        IWx2 = np.multiply(IWx,IWx)
        IWxy = np.multiply(IWx,IWy)
        IWy2 = np.multiply(IWy,IWy)
        IWx2_sum = np.sum(IWx2.ravel())
        IWxy_sum = np.sum(IWxy.ravel())
        IWy2_sum = np.sum(IWy2.ravel())
        H = np.array([[IWx2_sum,IWxy_sum],[IWxy_sum,IWy2_sum]])
        #Compute delta_p
        Hinv = np.linalg.inv(H)
        G = np.stack([np.multiply(IWx,err),np.multiply(IWy,err)],axis=-1)
        G = np.expand_dims(G,axis=3)
        G = np.matmul(Hinv,G)
        delta_p = np.sum(G,axis=(0,1)).reshape((2,))
        #Update parameter
        p = p + delta_p
        #Check if delta_p is smaller than threshold, break out of loop if true
        if np.linalg.norm(delta_p)<threshold:
            break
    return p