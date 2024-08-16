import numpy as np
import LucasKanadeAffine as LKA
import scipy.ndimage
import InverseCompositionAffine as ICA

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)
    M = LKA.LucasKanadeAffine(image1,image2)
    #M = ICA.InverseCompositionAffine(image1,image2)
    L = np.array([0.0,0.0,1.0])
    M = np.vstack([M,L])
    Minv = np.linalg.inv(M)
    image_transf = scipy.ndimage.affine_transform(image1,Minv)
    diff = np.abs(image_transf-image2)
    diff[image_transf==0] = 0
    mask = diff>0.2
    mask[0,:] = 0
    mask[:,0] = 0
    mask[image1.shape[0]-1,:]=0
    mask[:,image1.shape[1]-1]=0
    mask = scipy.ndimage.morphology.binary_dilation(mask)
    return mask
