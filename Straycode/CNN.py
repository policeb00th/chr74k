import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from  sklearn.model_selection import train_test_split
from labelConv import int2label

batch_size = 128
nb_classes = 62 # A-Z, a-z and 0-9
nb_epoch = 500

# Input image dimensions
img_rows, img_cols = 32, 32

# Path of data files
path = "/home/diptanshu/Documents/Char/Straycode"

# Load the preprocessed data and labels
X_train_all = np.load(path+"/trainPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")
Y_train_all = np.load(path+"/labelsPreproc.npy")

# Do multiple learnings and predictions with the aim of averaging them
for runID in range (18):    
    # Split in train and validation sets to get the "best" model.
    X_train, X_val, Y_train, Y_val = \
        train_test_split(X_train_all, Y_train_all, test_size=0.25, stratify=np.argmax(Y_train_all, axis=1))
                             
    # Parametrize the image augmentation class
    datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.15,
        height_shift_range = 0.15,
        shear_range = 0.4,
        zoom_range = 0.3,                    
        channel_shift_range = 0.1,
        channel_flip = True, # You must modify the ImageDataGenerator class for that parameter to work
        channel_flip_max = 1.) # You must modify the ImageDataGenerator class for that parameter to work

    ### CNN MODEL ###
    model = Sequential()

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation = 'relu', input_shape=(1, img_rows, img_cols)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, init='he_normal', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, init='he_normal', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='he_normal', activation = 'softmax'))

    ### LEARNING ###

    # First, use AdaDelta for some epochs because AdaMax gets stuck
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adadelta',  
                  metrics=["accuracy"])
                  
    # 20 epochs is sufficient
    model.fit(X_train, Y_train, batch_size=batch_size,
                        nb_epoch=20, 
                        validation_data=(X_val, Y_val),
                        verbose=1)
                      
    # Now, use AdaMax
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adamax',  
                  metrics=["accuracy"])
    
    # We want to keep the best model. This callback will store 
    # in a file the weights of the model with the highest validation accuracy  
    saveBestModel = ModelCheckpoint("best.kerasModelWeights", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

    # Make the model learn using the image generator
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=len(X_train),
                        nb_epoch=nb_epoch, 
                        validation_data=(X_val, Y_val),
                        callbacks=[saveBestModel],
                        verbose=1)

    ### PREDICTION ###
                        
    # Load the model with the highest validation accuracy
    model.load_weights("best.kerasModelWeights")

    # Load Kaggle test set
    X_test = np.load(path+"/testPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")

    # Predict the class (give the index in the one-hot vector of the most probable class)
    Y_test_pred = model.predict_classes(X_test)
    
    # Translate integers to character labels
    vInt2label = np.vectorize(int2label)
    Y_test_pred = vInt2label(Y_test_pred)
    
    # Save the predicitions in Kaggle format
    np.savetxt(path+"/CNN_pred_"+str(runID)+".csv", np.c_[range(6284,len(Y_test_pred)+6284),Y_test_pred], delimiter=',', header = 'ID,Class', comments = '', fmt='%s')
