# This script loads the USPS dataset (pickle file) and saves images to .png 
# format. It also performs dataset split into training and testing.
# Dependencies (WARNING): Python 2. If you use Python 3, you may have issues.

import numpy as np
import scipy
import os
import scipy.misc 
import pickle 
import gzip

def retrieve_uspsdataset():
    
    # Make directory, if not available. 
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    # Open dataset file.
    with gzip.open('usps_28x28.pkl', 'rb') as ifp:
        dataset = pickle.load(ifp, encoding='latin1')
    
    # Training and testing partitions.
    partition1 = dataset[0]
    partition2 = dataset[1]
    
    # Images and labels.
    partition1_x = partition1[0]
    partition1_y = partition1[1]
    partition2_x = partition2[0]
    partition2_y = partition2[1]
    
    # Concatenation of images' and labels' partitions.
    images = np.concatenate((partition1_x, partition2_x)) # 9298 x 1 x 28 x 28
    labels = np.concatenate((partition1_y, partition2_y)) # 9298
    
    images_x = np.transpose(images, [0, 2, 3, 1])
    
    # Reshape to 9298 x 28 x 28.
    images = np.reshape(images, [9298, 28, 28])
    
    # Dataset directory.
    rootPath = "./dataset/"
    #make_dir(rootPath)
    
    # Save all images to .png format to "all_images" folder.
    """allImagesPath = "usps_images/"
    make_dir(rootPath + allImagesPath)
    
    for num, image in enumerate(images):
        scipy.misc.imsave(rootPath + allImagesPath + 'sample' + str(num + 1) + ".png", image)
    
    make_dir("training-images/")
    make_dir("test-images/")
    """
    
    # Saves images as .png format based on class to respective folders.
    def image_class_specific_outputter(imgs, lbls, datasetType):
        classes = 10
    
        for c in range(classes):
            make_dir(datasetType + str(c))
        
        for num, image in enumerate(imgs):
            label = lbls[num]
            scipy.misc.imsave(datasetType + str(label) + "/sample" + str(num) + ".png", image)
    
    # Dataset split: 6,562 training, 729 validation, and 2,007 testing images.
    # Training: 7,291 (includes 729 images for validation)
    # Testing: 2,007
    x_train = images[:7291, :, :]
    y_train = labels[:7291]
    x_test = images[7291:, :, :]
    y_test = labels[7291:]
    
    #image_class_specific_outputter(X_train, Y_train, "training-images/")
    #image_class_specific_outputter(X_test, Y_test, "test-images/")
    return x_train, y_train, x_test, y_test