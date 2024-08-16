import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    h1,w1,d1 = im1.shape
    im2_ref1 = cv2.warpPerspective(im2,H2to1,(2*w1,h1))
    im1_ref1 = np.zeros((h1,2*w1,3))
    im1_ref1[:h1,:w1,:] = im1
    pano_im = np.maximum(im1_ref1,im2_ref1)
    cv2.imwrite('../results/6_10.jpg',im2_ref1)
    cv2.imwrite('../results/6_1.jpg',pano_im)
    np.save('../results/6_1.npy',H2to1)
    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    h1,w1,_ = im1.shape
    out_width = 2*w1
    h2,w2,_ = im2.shape
    x21_ref1 = np.matmul(H2to1,np.array([w2-1,0,1]).reshape((-1,1)))
    x21_ref1 = x21_ref1/x21_ref1[2]
    x22_ref1 = np.matmul(H2to1,np.array([w2-1,h2-1,1]).reshape((-1,1)))
    x22_ref1 = x22_ref1/x22_ref1[2]
    if(x21_ref1[0]>x22_ref1[0]):
        alpha = out_width/x21_ref1[0]
    else:
        alpha = out_width/x22_ref1[0]
    ty = int(-1*alpha*x21_ref1[1])
    M = np.zeros((3,3))
    M[0,0] = alpha
    M[1,1] = alpha
    M[1,2] = ty
    M[2,2] = 1
    out_height = int((alpha*x22_ref1[1])+ty)
    warp_im1 = cv2.warpPerspective(im1,M,(out_width,out_height))
    warp_im2 = cv2.warpPerspective(im2,np.matmul(M,H2to1),(out_width,out_height))
    pano_im = np.maximum(warp_im1,warp_im2)
    return pano_im

def generatePanorama(im1,im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1,desc2)
    H2to1 = ransacH(matches,locs1,locs2,num_iter=5000,tol=2)
    pano_im = imageStitching_noClip(im1,im2,H2to1)
    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching(im1,im2,H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img1 = im1[::2,::2,:]
    img2 = im2[::2,::2,:]
    panorama = generatePanorama(img1,img2)
    cv2.imwrite('../results/q6_3.jpg',panorama)
    cv2.imshow('panorama',panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()