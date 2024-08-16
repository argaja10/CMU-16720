import numpy as np
#import torchvision
import util
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io

if __name__ == '__main__':
    
    num_cores = util.get_num_CPU()
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    #path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)
    #util.save_filter_responses(filter_responses,'../q1.1.2.jpg')
    
    print("Computing dictionary")
    visual_words.compute_dictionary(num_workers=num_cores)
    
    dictionary = np.load('dictionary.npy')
    #img = visual_words.get_visual_words(image,dictionary)
    #util.save_wordmap(wordmap, filename)
    
    """
    #Visualizing and saving 3 wordmaps
    img1 = skimage.io.imread("../data/kitchen/sun_agfmsmojwptjoava.jpg")
    wordmap1 = visual_words.get_visual_words(img1,dictionary)
    util.save_wordmap(wordmap1,'../q1.3_1.jpg')
    img2 = skimage.io.imread("../data/kitchen/sun_aggjlwcdjsjcaemh.jpg")
    wordmap2 = visual_words.get_visual_words(img2,dictionary)
    util.save_wordmap(wordmap2,'../q1.3_2.jpg')
    img3 = skimage.io.imread("../data/kitchen/sun_ahyafiyoyrzkifbh.jpg")
    wordmap3 = visual_words.get_visual_words(img3,dictionary)
    util.save_wordmap(wordmap3,'../q1.3_3.jpg')
    """
    
    print("Building recognition system")
    visual_recog.build_recognition_system(num_workers=num_cores)
    print("Evaluating recognition system")
    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())
    
    """
    #HARD EXAMPLES TESTING
    visual_recog.predict_class("../data/laundromat/sun_azdzvpzrbrqcmssr.jpg")
    visual_recog.predict_class("../data/laundromat/sun_awytptirbthbwszg.jpg")
    visual_recog.predict_class("../data/highway/sun_bnxsatkjkhtpezcd.jpg")
    visual_recog.predict_class("../data/highway/sun_bgeixfvonqvlpwhg.jpg")
    visual_recog.predict_class("../data/desert/sun_bqujpdxcuslyzejj.jpg")
    """
