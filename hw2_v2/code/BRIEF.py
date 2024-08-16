import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    #############################
    #Choice of test pairs is done based on method II as mentioned in Michael Calonder's BRIEF paper
    #Method II refers to i.i.d Gaussian sampling within the patch
    compareX = np.zeros((nbits,))
    compareY = np.zeros((nbits,))
    center_pix_id = np.floor((patch_width-1)/2)
    mean = np.array([0,0])+center_pix_id
    cov = ((patch_width*patch_width)/25)*np.array([[1.0,0.0],[0.0,1.0]])
    X = np.random.multivariate_normal(mean,cov,nbits)
    Y = np.random.multivariate_normal(mean,cov,nbits)
    X = X.astype(int)
    Y = Y.astype(int)
    #Change X, Y values if they are out of range
    for i in range(nbits):
        if(X[i,0]<0):
            X[i,0] = 0
        if(X[i,0]>patch_width-1):
            X[i,0] = patch_width-1
        if(X[i,1]<0):
            X[i,1] = 0
        if(X[i,1]>patch_width-1):
            X[i,1] = patch_width-1
        if(Y[i,0]<0):
            Y[i,0] = 0
        if(Y[i,0]>patch_width-1):
            Y[i,0] = patch_width-1
        if(Y[i,1]<0):
            Y[i,1] = 0
        if(Y[i,1]>patch_width-1):
            Y[i,1] = patch_width-1
        compareX[i] = (X[i,0]*patch_width)+X[i,1]
        compareY[i] = (Y[i,0]*patch_width)+Y[i,1]
    np.save('../results/testPattern.npy',[compareX,compareY])
    return  compareX, compareY

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG, compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    desc = []
    locs = []
    patch_width = 9
    c = int(np.floor((patch_width-1)/2))
    n = compareX.shape[0]
    nbits = int(n)
    M = locsDoG.shape[0]
    #Iterate over all keypoints. If keypoint can produce a valid descriptor, then
    #compute the pixels to be compared from the linear indices, and make the bit pattern
    for j in range(M):
        if(valid_desc_point(locsDoG[j,:],patch_width,im.shape)):
            bit_vec = np.zeros((nbits,))
            for i in range(n): 
                #Compute pixel coordinates based on linear index values
                X = compareX[i]
                Y = compareY[i]
                X_i = int(np.floor(X/patch_width))
                X_j = int(X-(X_i*patch_width))
                Y_i = int(np.floor(Y/patch_width))
                Y_j = int(Y-(Y_i*patch_width))
                #Compare the two pixel values in the Gaussian pyramid
                comp = gaussian_pyramid[locsDoG[j,1]+Y_i-c,locsDoG[j,0]+Y_j-c,locsDoG[j,2]] - \
                gaussian_pyramid[locsDoG[j,1]+X_i-c,locsDoG[j,0]+X_j-c,locsDoG[j,2]]
                if(comp>0):
                    bit_vec[i] = 1
            desc.append(bit_vec)
            locs.append(locsDoG[j,:])
    desc = np.stack(desc,axis=-1)
    locs = np.stack(locs,axis=-1)
    return locs.T, desc.T

def valid_desc_point(locsDoG_pt,patch_width,img_dim):
    #Checks if the keypoint can produce a valid descriptor
    #A keypoint can produce a valid descriptor if it is sufficiently within the edges
    h = img_dim[0]
    w = img_dim[1]
    delta = int((patch_width-1)/2)
    i = locsDoG_pt[1]
    j = locsDoG_pt[0]
    if(i-delta<0 or i+delta>=h):
        return False
    if(j-delta<0 or j+delta>=w):
        return False
    return True

def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    locsDoG, gaussian_pyramid = DoGdetector(im)
    compareX, compareY = np.load('../results/testPattern.npy')
    locs, desc = computeBrief(im,gaussian_pyramid,locsDoG,compareX,compareY)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    #plt.savefig('../results/matches_chickenbroth.jpg')
    plt.show()

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.show()
    #plt.draw()
    #plt.waitforbuttonpress(0)
    #plt.close(fig)
    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
