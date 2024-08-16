import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    #Using X1, X2, construct matrix A (dimensions 2N x 9)
    N = p1.shape[1]
    A = np.zeros((2*N,9))
    A[::2,0] = p2[0,:]
    A[::2,1] = p2[1,:]
    A[::2,2] = 1
    A[1::2,3] = p2[0,:]
    A[1::2,4] = p2[1,:]
    A[1::2,5] = 1
    p4 = -1*np.multiply(p2[0,:],p1[0,:])
    p5 = -1*np.multiply(p2[0,:],p1[1,:])
    p6 = -1*np.multiply(p2[1,:],p1[0,:])
    p7 = -1*np.multiply(p2[1,:],p1[1,:])
    A[::2,6] = p4
    A[1::2,6] = p5
    A[::2,7] = p6
    A[1::2,7] = p7
    A[::2,8] = -1*p1[0,:]
    A[1::2,8] = -1*p1[1,:]
    u,_,_ = np.linalg.svd(np.matmul(A.T,A),full_matrices=False)
    h = u[:,8]
    H2to1 = np.reshape(h,(3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    N = matches.shape[0]
    for i in range(num_iter):
        p_set = np.random.permutation(np.arange(N))
        min_set = p_set[:4]
        X1 = locs1[matches[min_set,0],:2]
        X2 = locs2[matches[min_set,1],:2]
        H = computeH(X1.T,X2.T)
        num_inliers = 0
        for j in range(4,N,1):
            x2 = locs2[matches[p_set[j],1],:2]
            x1 = locs1[matches[p_set[j],0],:2]
            x1 = np.array([x1[0],x1[1],1]).reshape((-1,1))
            x1_pred = np.matmul(H,np.array([x2[0],x2[1],1]).reshape((-1,1)))
            x1_pred = x1_pred/x1_pred[2]
            if(np.linalg.norm(x1-x1_pred)<tol):
                num_inliers = num_inliers+1
        if(i==0):
            bestH = H
            bestH_num_inliers = num_inliers
        else:
            if(num_inliers>bestH_num_inliers):
                bestH = H
                bestH_num_inliers = num_inliers
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_L.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    Hest = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

