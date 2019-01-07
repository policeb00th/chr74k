import os
import glob
import pandas as pd
import math

import numpy as np
from scipy.misc import imread, imsave, imresize
from natsort import natsorted

from labelConv import label2int

# Path of data files
path = "/home/diptanshu/Documents/Char/Straycode"

# Input image dimensions
img_rows, img_cols = 32, 32

# Keep or not the initial image aspect ratio
keepRatio = False

# Suffix of the created directories and files
suffix = "Preproc_" + str(img_rows) + "_"+ str(img_cols) + ("_kr" if keepRatio else "")

# Create the directories if needed
if not os.path.exists( path + "/train"+suffix ):
    os.makedirs(path + "/train"+suffix)
if not os.path.exists( path + "/test"+suffix ):
    os.makedirs(path + "/test"+suffix)
    
    
### Images preprocessing ###

for setType in ["train", "test"]:
    # We have to make sure files are sorted according to labels, even if they don't have trailing zeros
    files = natsorted(glob.glob(path + "/"+setType+"/*"))
    
    data = np.zeros((len(files), img_rows, img_cols)) #will add the channel dimension later
    
    for i, filepath in enumerate(files):
        image = imread(filepath, True) #True: flatten to grayscale
        if keepRatio:
            # Find the largest dimension (height or width)
            maxSize = max(image.shape[0], image.shape[1])
            
            # Size of the resized image, keeping aspect ratio
            imageWidth = math.floor(img_rows*image.shape[0]/maxSize)
            imageHeigh = math.floor(img_cols*image.shape[1]/maxSize)
            
            # Compute deltas to center image (should be 0 for the largest dimension)
            dRows = (img_rows-imageWidth)//2
            dCols = (img_cols-imageHeigh)//2
                        
            imageResized = np.zeros((img_rows, img_cols))
            imageResized[dRows:dRows+imageWidth, dCols:dCols+imageHeigh] = imresize(image, (imageWidth, imageHeigh))
            
            # Fill the empty image with the median value of the border pixels
            # This value should be close to the background color
            val = np.median(np.append(imageResized[dRows,:],
                                      (imageResized[dRows+imageWidth-1,:],
                                      imageResized[:,dCols],
                                      imageResized[:,dCols+imageHeigh-1])))
                                      
            # If rows were left blank
            if(dRows>0):
                imageResized[0:dRows,:].fill(val)
                imageResized[dRows+imageWidth:,:].fill(val)
                
            # If columns were left blank
            if(dCols>0):
                imageResized[:,0:dCols].fill(val)
                imageResized[:,dCols+imageHeigh:].fill(val)
        else:
            imageResized = imresize(image, (img_rows, img_cols))
        
        # Add the resized image to the dataset
        data[i] = imageResized
        
        #Save image (mostly for visualization)
        filename = filepath.split("\\")[-1]
        filenameDotSplit = filename.split(".")
        newFilename = str(int(filenameDotSplit[0])).zfill(5) + "." + filenameDotSplit[-1].lower()  #Add trailing zeros
        newName = "/".join(filepath.split("\\")[:-1] ) +suffix+ "/" + newFilename
        imsave(newName, imageResized)
        
    # Add channel/filter dimension
    data = data[:,np.newaxis,:,:] 
    
    # Makes values floats between 0 and 1 (gives better results for neural nets)
    data = data.astype('float32')
    data /= 255
    
    # Save the data as numpy file for faster loading
    np.save(path+"/"+setType+suffix+".npy", data)

    
### Labels preprocessing ###

# Load labels
y_train = pd.read_csv(path+"/trainLabels.csv").values[:,1] #Keep only label

# Convert labels to one-hot vectors
Y_train = np.zeros((y_train.shape[0], len(np.unique(y_train))))

for i in range(y_train.shape[0]):
    Y_train[i][label2int(y_train[i])] = 1 # One-hot

# Save preprocessed label to nupy file for faster loading
np.save(path+"/"+"labelsPreproc.npy", Y_train)

