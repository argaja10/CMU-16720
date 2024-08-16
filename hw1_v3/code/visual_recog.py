import numpy as np
#import threading
import multiprocessing
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import sklearn.metrics as skm
    
def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    #features = Parallel(n_jobs=num_workers)(delayed(get_image_feature)\
    #                    (i,dictionary,2,K) for i in train_data.f.files)
    p = multiprocessing.Pool(num_workers)
    features = p.starmap(get_image_feature,[(filename,dictionary,2,200) for filename in train_data['files']])
    features = np.vstack(features)
    np.savez_compressed("trained_system.npz",features=features,\
                        labels=train_data['labels'],dictionary=dictionary,SPM_layer_num=2)
    
def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    train_data_features = trained_system['features']
    train_data_labels = trained_system['labels']
    d = trained_system['dictionary']
    test_data_labels_true = test_data['labels']
    test_data_labels_pred = -1*np.ones(test_data_labels_true.shape)
    #test_data_labels_pred = Parallel(n_jobs=num_workers)\
    #(delayed(evaluate_test_img)(i,filename) for i,filename in enumerate(test_data.f.files))
    p = multiprocessing.Pool(num_workers)
    test_data_labels_pred = p.starmap(evaluate_test_img,[(i,filename,d,train_data_features,train_data_labels) for i,filename in enumerate(test_data['files'])])
    test_data_labels_pred = np.asarray(test_data_labels_pred)
    
    def compute_confusion_matrix(true_labels,pred_labels,num_classes):
        conf = np.zeros((8,8))
        n = true_labels.shape[0]
        for i in range(n):
            conf[true_labels[i],pred_labels[i]] = conf[true_labels[i],pred_labels[i]] + 1
        acc = np.trace(conf)/n
        return conf.astype(int), acc
    
    conf, accuracy = compute_confusion_matrix(test_data_labels_true,test_data_labels_pred,8)
    #print(skm.confusion_matrix(test_data_labels_true,test_data_labels_pred))
    #print(skm.accuracy_score(test_data_labels_true,test_data_labels_pred))
    return conf, accuracy

def evaluate_test_img(idx,filename,d,train_data_features,train_data_labels):
    test_feature = get_image_feature(filename,d,2,200)
    dists = distance_to_set(test_feature,train_data_features)
    return train_data_labels[np.argmax(dists)]

def predict_class(img_filename):
    trained_system = np.load("trained_system.npz")
    train_data_features = trained_system['features']
    train_data_labels = trained_system['labels']
    d = trained_system['dictionary']
    label_names = ['aquarium','park','desert','highway','kitchen','laundromat','waterfall','windmill']
    print(label_names[evaluate_test_img(0,img_filename,d,train_data_features,train_data_labels)])
    

def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    img = skimage.io.imread("../data/"+file_path)
    wordmap = visual_words.get_visual_words(img,dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    return feature

def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    sim = np.sum(np.minimum(word_hist,histograms),axis=1)
    return sim

def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist,_ = np.histogram(np.ravel(wordmap),bins=dict_size,range=(0,dict_size))
    return hist/(np.sum(hist))

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '
    lvl1_split = quad_split(wordmap)    #2x2 split of wordmap (level 1)
    lvl1_cell_hists = []
    lvl2_cell_hists = []
    
    for sub_blck in lvl1_split:
        lvl2_cell = quad_split(sub_blck)    #2x2 split of a level-1 sub-block (one cell in level 2)
        h0_2 = get_feature_from_wordmap(lvl2_cell[0],dict_size)
        h1_2 = get_feature_from_wordmap(lvl2_cell[1],dict_size)
        h2_2 = get_feature_from_wordmap(lvl2_cell[2],dict_size)
        h3_2 = get_feature_from_wordmap(lvl2_cell[3],dict_size)
        lvl2_cell_hists.extend((h0_2,h1_2,h2_2,h3_2))
        h_1 = h0_2+h1_2+h2_2+h3_2
        h_1 = h_1/(np.sum(h_1))
        lvl1_cell_hists.append(h_1)
    '''
    wordmap_cells = [wordmap]
    for i in range(layer_num):
        wordmap_cells = quad_split(wordmap_cells)
    #Compute all histograms for the last level in SPM
    hist_cells = []
    for cell in wordmap_cells:
        hist_cells.append(get_feature_from_wordmap(cell,dict_size))
    wt = 0.5
    hist_all = [(wt*np.hstack(hist_cells))]
    for i in range(layer_num):
        if i!=(layer_num-1):
            wt = wt/2
        new_hist_cells = []
        n = len(hist_cells)
        for j in range(0,n,4):
            merge_cells_hist = sum(hist_cells[j:j+4])
            merge_cells_hist = merge_cells_hist/(np.sum(merge_cells_hist))
            new_hist_cells.append(merge_cells_hist)
        hist_cells = new_hist_cells
        hist_all.insert(0,wt*np.hstack(hist_cells))
    """
    lvl0_cell_hist = sum(lvl1_cell_hists)
    l0 = lvl0_cell_hist/(np.sum(lvl0_cell_hist))
    l1 = np.hstack(lvl1_cell_hists)
    l2 = np.hstack(lvl2_cell_hists)
    hist_all = np.hstack(((0.25)*l0,(0.25)*l1,(0.5)*l2))"""
    hist_all = np.hstack(hist_all)
    hist_all = hist_all/(np.sum(hist_all))
    return hist_all

def quad_split(mat_list):
    '''
    Splits every matrix in mat_list into 4 sub-blocks. Ignores last row/column if dimension values are odd-valued
    
    [input]
    * mat_list: numpy.ndarray of shape (H,W)
    
    [output]
    * mat_blocks_list: list of 4 numpy.ndarrays corresponding to top-left, top-right, bottom-left, bottom-right
    sub matrices of every matrix in mat_list
    '''
    #mat_blocks = np.array_split(np.array_split(mat,2,axis=0)[0],2,axis=1)+\
    #np.array_split(np.array_split(mat,2,axis=0)[1],2,axis=1)
    #return mat_blocks
    mat_blocks_list = []
    for mat in mat_list:
        h,w = mat.shape
        blk_0 = mat[0:int(h/2),0:int(w/2)]
        blk_1 = mat[0:int(h/2),int(w/2):2*int(w/2)]
        blk_2 = mat[int(h/2):2*int(h/2),0:int(w/2)]
        blk_3 = mat[int(h/2):2*int(h/2),int(w/2):2*int(w/2)]
        mat_blocks_list.extend((blk_0,blk_1,blk_2,blk_3))
    return mat_blocks_list

    

