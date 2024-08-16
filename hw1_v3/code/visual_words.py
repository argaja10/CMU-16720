import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    if len(image.shape) == 2:
        image = np.tile(image[:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    fltr_resp = extract_filter_responses(image)
    h,w,d = fltr_resp.shape
    fltr_resp = np.reshape(fltr_resp,(h*w,d))
    dists = scipy.spatial.distance.cdist(fltr_resp,dictionary)
    wordmap = np.argmin(dists,axis=1)
    wordmap = np.reshape(wordmap,(h,w))
    return wordmap

def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    i,alpha,image_path = args
    img = skimage.io.imread("../data/"+image_path)
    img = img.astype('float')/255
    fltr_bank_resp = extract_filter_responses(img)
    h,w,d = fltr_bank_resp.shape
    fltr_bank_resp = np.reshape(fltr_bank_resp,(h*w,d))
    return fltr_bank_resp[np.random.choice(h*w,alpha),:]

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    alpha = 100
    train_data = np.load("../data/train_data.npz")
    #fb_resp_rndsamp_unv = Parallel(n_jobs=num_workers)\
    #(delayed(compute_dictionary_one_image)([i,alpha,filename]) for i,filename in enumerate(train_data.f.files))
    p = multiprocessing.Pool(num_workers)
    args_list = [(i,alpha,filename) for i,filename in enumerate(train_data.f.files)]
    fb_resp_rndsamp_unv = p.map(compute_dictionary_one_image,args_list)
    fb_resp_rndsamp_unv = np.vstack(fb_resp_rndsamp_unv)
    print("Performing KMeans")
    kmeans = sklearn.cluster.KMeans(n_clusters=200,n_jobs=num_workers).fit(fb_resp_rndsamp_unv)
    print("KMeans done!")
    dictionary = kmeans.cluster_centers_
    np.save("dictionary.npy",dictionary)

