import numpy as np
import cv2

def createGaussianPyramid(im,sigma0=1,k=np.sqrt(2),levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
    return

def displayImage_keypoints(image, keypoints):
    for p in keypoints:
        x =p[0]
        y = p[1]
        cv2.circle(image,(x,y),2,(0,255,0),-1)
    cv2.imwrite('../results/keypoints_q1.jpg',image)
    cv2.imshow('Image with keypoints',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def createDoGPyramid(gaussian_pyramid,levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    DoG_levels = levels[1:]
    DoG_numlevels = len(levels)-1
    for i in range(DoG_numlevels):
    	DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid,axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    eps = 1e-6
    principal_curvature = []
    h,w,d = DoG_pyramid.shape
    for i in range(d):
    	Dxx = cv2.Sobel(DoG_pyramid[:,:,i],-1,2,0,ksize=3)
    	Dxy = cv2.Sobel(DoG_pyramid[:,:,i],-1,1,1,ksize=3)
    	Dyy = cv2.Sobel(DoG_pyramid[:,:,i],-1,0,2,ksize=3)
    	trH_sqr = np.multiply(Dxx+Dyy,Dxx+Dyy)
    	detH = np.multiply(Dxx,Dyy)-np.multiply(Dxy,Dxy)
    	R = trH_sqr/(detH+eps)
    	principal_curvature.append(R)
    principal_curvature = np.stack(principal_curvature,axis=-1)
    return principal_curvature

def getLocalExtrema(DoG_pyramid,DoG_levels,principal_curvature,th_contrast=0.03,th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    h,w,d = DoG_pyramid.shape
    #Filter out points that do not satisfy |D(x,y,sigma)|>th_contrast
    mask = np.logical_and(np.abs(DoG_pyramid)>th_contrast,principal_curvature<th_r)
    imp_pixels = np.argwhere(mask)
    for pixel in imp_pixels:
    	if(check_if_keypoint(pixel,DoG_pyramid)):
            #Note x-coordinate is column number and y-coordinate is row number
            #Storing in locsDoG in (x,y,sigma) format
    		locsDoG.append(np.array([pixel[1],pixel[0],pixel[2]]))
    locsDoG = np.stack(locsDoG,axis=-1)
    return locsDoG.T
    
def check_if_keypoint(pixel,Dp):
    '''
    Checks if a given pixel is a local maxima and satisfies the principal curvature constraints
    
    Given a pixel (x,y,sigma), the function returns True if:
        D(x,y,sigma) is a local extrema in both scale and space
        PC(x,y,sigma)<theta_r
    Else the function returns False
    '''
    imH,imW,num_scales = Dp.shape
    cond1 = pixel[0]==0 or pixel[0]==imH-1
    cond2 = pixel[1]==0 or pixel[1]==imW-1
    cond3 = pixel[2]==0 or pixel[2]==num_scales-1
    if not(cond1 or cond2 or cond3):
        i = pixel[0]
        j = pixel[1]
        s = pixel[2]
        if(Dp[i,j,s]>Dp[i,j,s-1] and Dp[i,j,s]>Dp[i,j,s+1]):
            nbhd = Dp[i-1:i+2,j-1:j+2,s]
            cmax = np.argmax(nbhd)
            cmin = np.argmin(nbhd)
            if(cmax==4 or cmin==4):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def DoGdetector(im,sigma0=1,k=np.sqrt(2),levels=[-1,0,1,2,3,4],th_contrast=0.03,th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    im_pyr = createGaussianPyramid(im,sigma0=1,k=np.sqrt(2),levels=[-1,0,1,2,3,4,5])
    dgp, dgl = createDoGPyramid(im_pyr,levels)
    pc = computePrincipalCurvature(dgp)
    locsDoG = getLocalExtrema(dgp,dgl,pc,th_contrast,th_r)
    return locsDoG, im_pyr

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr,DoG_levels,pc_curvature,th_contrast,th_r)
    displayImage_keypoints(im.copy(),locsDoG)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

