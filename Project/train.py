##############################################
# Student information: Hany Hamed
# Group: BS18-ROB
# Project Contest IML course Fall 2020
# This code is the main file for training
# Note for credits: I would like to thank my collouge Yusuf Mesbah (@Yusufroshdy) for his hints, insights and his help for some of the augmentations and optimization for CNN architechtures
##############################################
# -*- coding: utf-8 -*-

"""# Test history and documentation
*** Note with Adam model1_1  acc=0.9883
* model1_1 (M1): accuracy: 0.9914
* model1_1_1(M1): the same as model1_1 but with NADAM(1e-4) accuracy: 0.9886
* model1_2 (M3): 5 conv, 1fc, accuracy: 0.9907
* model1_3 (M3): but with 2nd augmentation, accuracy: 0.9206
* model1_4 (M3): but with 2nd augmentation with some changes, accuracy: 0.9887
* model1_5 (M3): but with 2nd augmentation with some changes, accuracy: 0.9858
* model1_6 (M4): with the same augmentation of model1_5 accuracy: 0.9839
* model1_7 (M4): with 1st augmentation, accuracy: 0.9858 (not saved)

* model1_8 (M1): with 1st augmentation, accuracy: 0.9896
* model1_9 (M1): with 2nd augmentation and remove lr secheduling, accuracy: 0.9836
* model1_10 (M2): with 2nd augmentation and remove lr sch. accuracy: 0.9822
* model1_11 (M2): without augmentation (overfit), accuracy(test)= 0.9945
* model1_12 (M4): without augmentation (overfit), accuracy: 0.9930
* model1_13 (M5): accuracy: without augmentation 0.9938
* model1_14 (M5): accuracy: with augmentation 0.9952 (Too big model~1GB)
* model1_15 (M7): ????
* model1_16 (M7): with flipping bits accuracy:0.992
* model1_17 (M8): with normal augmented data
* model1_18 (M8): with changes in the augmentations with some changes in the model like depth of the model
* model1_19 (M8): with changes in the augmentations (brightness) and some changes in the model (num. filters, BN) (Test)


-----------------------------------------------------------------------
* model_gan_1 (M2): (GAN Data) with Aug. accuracy: 0.9995 bad on testing
* model_gan_2 (M2): with Aug. accuracy: 0.9965 (GAN data + original data)
* model_gan_3 (M8): GAN data with Aug. from model1_19 (0.58)

-----------------------------------------------------------------------
* model5_1 (M8): New augmentations accuracy: 0.992  test_acc: 0.882
* model5_2 (M9 -- increased dropout and one fc): inverted augmentations and increased the probs, accuracy: 0.9948, test_acc: 0.887
* model5_3 (M9_2): The same with augmentation in model5_2 but with regularization to the model9 and learnable max pooling (not better than before)
* model5_4 (M9): with the new augmentations, accuracy: 1.0000, accuracy(local_test): 0.9746, (codalab_test): 0.85
* model5_5 (M9): with new augmentations,  accuracy: 1.0000, accuracy(local_test): 0.9746 
* model5_6 (M9_2): 10epochs,  accuracy: 1.0000, accuracy(local_test): 0.9746 
* model5_7 (M9_3): 10epochs,  accuracy: 0.9924, accuracy(local_test): 0.9694

* model5_9 (M9): augmentations with imgaug ,accuracy: 0.9925, test:0.85
* model5_10(M9): augmentation flow from call to the other not the same stored data 
* model5_11 (M9): add bounding box augmentations training accuracy: 0.9942 - val_accuracy: 0.9958 - testing_local accuracy: 0.9948
* model5_12 (M9): use aug3 + augmentations with increased prob than 5_11, accuracy: 98.8
* model5_13 (M9 with Adam): accuracy: 0.9908 - val_loss: 0.0304 val_accuracy: 0.9914 test_loc_accuracy: 0.9232
* model5_14: continue of training for model5_13 with more augmentation
* model5_15: continue of training for model5_12 with same augmentation as model5_14
* model5_16: continue of training for model5_15 without change
* model5_17: continue 5_16 with some chagnges in the augmentation
* model5_18: continue 5_17 with some chagnges in the augmentation and use aug_gen2 data 
* model5_19: continue of 5_18 without early stopping with 30 epoch (worse than 18)
* model5_20: continue of 5_18 with early stopping
* model5_21: continue of model5_18 with x_aug_new2 data
* model5_22: continue of model5_18 with aug2 with some changes in the augmentations (without bounding box)
* model5_23 (M9): aug_new2 dataset with pattern augmentation
* model5_24 (M9): aug2 dataset with pattern augmentation and bounding box (It was better than 23)
* model5_25 (M9_4): aug3 dataset with pattern and bounding box, accuracy: 0.9891 and the best from one training loop
* model5_26 (M9_4): salt_pepper augmentation with usual augmentation (bad in test and colab but good in validation and training)
* model5_27 (M9_4): (Gives on colab 88.9) salt_pepper augmentation + self implemented augmentation + imgaug augmentation + grayscale image backgroun augmentation
-----------------------------------------------------------------------
-- Papers:
* [Improvement of Generalization Ability of
Deep CNN via Implicit Regularization in
Two-Stage Training Process](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8306949)
* [A Generalization of Convolutional Neural
Networks to Graph-Structured Data](https://arxiv.org/pdf/1704.08165.pdf)
* [Multi-column Deep Neural Networks for Image Classification](https://arxiv.org/pdf/1202.2745.pdf)
* [Best Practices for Convolutional Neural Networks
Applied to Visual Document Analysis](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Simard.pdf)
* [A survey on Image Data Augmentation for Deep Learning
](https://link.springer.com/article/10.1186/s40537-019-0197-0)
"""

"""ML_Project_core1_(2)_(1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bTs6sz38gdlW-SWuBvFMeGGONeYcvLpp

# Install libraies
"""

"""# 1. Import libraries"""
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LayerNormalization, Input, concatenate, Average, GlobalMaxPool2D, Activation
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image as pImage
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import imgaug.augmenters as iaa
np.random.seed(123)

"""# 2. Import the dataset

## Only colab
"""
# path = ""
# X1, Y1 = np.load(path+"x.npy"), np.load(path+"y.npy")
# X, Y = X1, Y1
# print(X.shape, Y.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=123)

"""## Augmented by imgaug Colab data"""
path = ""
postfix = f"2" #3 #f"_new{2}" #2 #3 
X1, Y1 = np.load(path+f"x_aug{postfix}.npy"), np.load(path+f"y_aug{postfix}.npy")
X, Y = X1, Y1
print(X.shape, Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=123)

"""## Original mnist"""
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

"""## With only GAN"""
# path = "gdrive/My Drive/acgan/gen/"
# X2, Y2 = np.load(path+"x_gen_data.npy"), np.load(path+"y_gen_data.npy")
# X, Y = X2, Y2
# print(X.shape, Y.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=123)

"""## GAN with colab MNIST"""
# path = "gdrive/My Drive/"
# X1, Y1 = np.load(path+"x.npy"), np.load(path+"y.npy")
# path = "gdrive/My Drive/acgan/gen/"
# X2, Y2 = np.load(path+"x_gen_data.npy"), np.load(path+"y_gen_data.npy")
# X, Y = np.vstack([X1,X2]), np.hstack([Y1,Y2])
# print(X.shape, Y.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=123)

"""## With flipping image + colab mnist"""
# path = "gdrive/My Drive/"
# X1, Y1 = np.load(path+"x.npy"), np.load(path+"y.npy")
# X2 = np.array([])
# X2 = np.array(abs((X1) - np.full((len(X1),28,28,1),255)))
# X, Y = np.vstack([X1,X2]), np.hstack([Y1,Y1])
# print(X.shape, Y.shape)
# # print(X2.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True, random_state=123)

"""# 3. Display part of the dataset"""
# num_rows, num_cols = 2, 10
# f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
#                      gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
#                      squeeze=True)
# for r in range(num_rows):
#     for c in range(num_cols):
#         image_index = r * num_cols + c
#         ax[r,c].axis("off")
#         ax[r,c].imshow(x_train[image_index].squeeze(), cmap='gray')
#         # ax[r,c].set_title('No. %d' % y_train[image_index])
# plt.show()

"""# 4. Models architechtures"""
def make_model1():
    m = Sequential()

    m.add(Conv2D(filters=256, kernel_size=5, 
                 input_shape=(28,28,1), activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))

    m.add(Dense(10, activation="softmax"))


    return m

def make_model2():
    m = Sequential()

    m.add(Conv2D(filters=512, kernel_size=5, padding="same",
                 input_shape=(28,28,1), activation='relu'))

    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())


    m.add(Conv2D(filters=256, kernel_size=5, padding="same",
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=5, padding="same",
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))

    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))

    m.add(Dense(10, activation="softmax"))


    return m

def make_model3():
    m = Sequential()

    m.add(Conv2D(filters=128, 
                 kernel_size=3, 
                 activation='relu',
                 input_shape=(28,28,1)))
    m.add(Conv2D(filters=128, 
              kernel_size=3, 
              activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())


    m.add(Conv2D(filters=256, 
                 kernel_size=3, 
                 activation='relu'))
    m.add(Conv2D(filters=256, 
              kernel_size=3, 
              activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=512, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(Dropout(0.2))
  
    m.add(Dense(10, activation="softmax"))


    return m

def make_model4():
    m = Sequential()

    m.add(Conv2D(filters=256, kernel_size=5, 
                 input_shape=(28,28,1), activation='relu'))
    m.add(Conv2D(filters=256, kernel_size=5, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))

    m.add(Dense(10, activation="softmax"))


    return m

def make_model5():
    m = Sequential()

    m.add(Conv2D(filters=64, kernel_size=3, padding="same", 
                 input_shape=(28,28,1), activation='relu'))
    m.add(Conv2D(filters=64, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m

def make_model6():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=3, padding="same", 
                 input_shape=(28,28,1), activation='relu'))
    m.add(Conv2D(filters=32, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=5, padding="same", 
                 activation='relu'))
    m.add(Conv2D(filters=64, kernel_size=5, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(512, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m

# simpler variant of 5
def make_model7():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=3, padding="same", 
                 input_shape=(28,28,1), activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=32, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=5, padding="same", 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2, strides=1))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(128, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m

# simpler variant of 5
def make_model8():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=3, 
                 input_shape=(28,28,1), activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=2, 
                 activation='relu'))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))

    m.add(Dense(10, activation="softmax"))


    return m

# simpler variant of 5
def make_model9():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=3, 
                 input_shape=(28,28,1), activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=2, 
                 activation='relu'))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m

# Variant of 9 with regularization and learnable pooling layers(conv with stride 2)
def make_model9_2():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=3, 
                 input_shape=(28,28,1), activation='relu', kernel_regularizer=l1(0.01)))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, kernel_regularizer=l1(0.01), 
                 activation='relu'))
    m.add(Conv2D(filters=64, kernel_size=2, strides=2, 
                 activation='relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=3, kernel_regularizer=l1(0.01), 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, kernel_regularizer=l1(0.01), 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=2, strides=2, 
                 activation='relu'))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=128, kernel_size=3, kernel_regularizer=l1(0.01), 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=2, kernel_regularizer=l1(0.01), 
                 activation='relu'))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m

# simpler variant of 9
def make_model9_3():
    m = Sequential()

    # m.add(Conv2D(filters=32, kernel_size=3, 
    #              input_shape=(28,28,1), activation='relu'))
    # m.add(BatchNormalization())
    # m.add(Conv2D(filters=64, kernel_size=3, 
    #              activation='relu'))
    # m.add(BatchNormalization())
    # m.add(Conv2D(filters=64, kernel_size=3, 
    #              activation='relu'))
    # m.add(BatchNormalization())
    # m.add(MaxPool2D(pool_size=2))
    # m.add(BatchNormalization())

    # m.add(Conv2D(filters=128, kernel_size=3, 
    #              activation='relu'))
    # m.add(BatchNormalization())
    # m.add(Conv2D(filters=128, kernel_size=3, 
    #              activation='relu'))  # 128
    # m.add(BatchNormalization())
    # m.add(MaxPool2D(pool_size=2))
    # m.add(BatchNormalization())

    m.add(Conv2D(filters=64, kernel_size=3,
                 activation='relu'))
    # m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())


    # m.add(Conv2D(filters=512, kernel_size=4, 
    #              activation='relu'))
    # m.add(BatchNormalization())
    # m.add(Conv2D(filters=128, kernel_size=2, 
    #              activation='relu'))
    # m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    # m.add(Dense(256, activation="relu"))
    # m.add(BatchNormalization())
    # m.add(Dropout(0.4))

    m.add(Dense(10, activation="softmax"))


    return m

# simpler variant of 5
def make_model9_4():
    m = Sequential()

    m.add(Conv2D(filters=64, kernel_size=3, 
                 input_shape=(28,28,1), activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=3, 
                 activation='relu'))
    m.add(MaxPool2D(pool_size=2))
    m.add(BatchNormalization())

    # m.add(Conv2D(filters=256, kernel_size=5, padding="same", 
    #              activation='relu'))
    m.add(Conv2D(filters=256, kernel_size=3, 
                 activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=256, kernel_size=2, 
                 activation='relu'))
    m.add(BatchNormalization())

    m.add(Flatten())
    
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))
    m.add(Dense(256, activation="relu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.3))

    m.add(Dense(10, activation="softmax"))


    return m
"""# 5. Callbacks (Augmentations, early stopping, tensorboard, ...etc)"""

# Source: https://www.kaggle.com/fraluc/cnn-keras-with-custom-data-augmentation-99-52

# Gaussian Blur
def gaussian_blur(X, sigma=0.5):
  return gaussian_filter(X,sigma)

# Gaussian Noise -- Salt an pepper noise
# Source: https://gist.github.com/Prasad9/077050d9a63df17cb1eaf33df4158b19
def gaussian_noise(X):
  # Need to produce a copy as to not modify the original image
  X_img = X
  row, col, _ = X_img.shape
  salt_vs_pepper = 0.3
  amount = 0.05
  num_salt = np.ceil(amount * X_img.size * salt_vs_pepper)
  num_pepper = np.ceil(amount * X_img.size * (1.0 - salt_vs_pepper))

  # Add Salt noise
  coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape[:2]]
  X_img[coords[0], coords[1], :] = 255

  # Add Pepper noise
  coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape[:2]]
  X_img[coords[0], coords[1], :] = 0
  return X_img

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops
# Dilation
def dilation(X):
  return cv2.dilate(X,np.ones((2,2),np.uint8),iterations = 1).reshape(28,28,1)

# erosion
def erosion(X):
  return cv2.erode(X,np.ones((2,2),np.uint8),iterations = 1).reshape(28,28,1)

# opening
def opening(X):
  return cv2.morphologyEx(X, cv2.MORPH_OPEN, np.ones((2,2),np.uint8),iterations = 1).reshape(28,28,1)

# closing
def closing(X):
  return cv2.morphologyEx(X, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8),iterations = 1).reshape(28,28,1)

def gradient(X):
  return cv2.morphologyEx(X, cv2.MORPH_GRADIENT, np.ones((2,2),np.uint8),iterations = 1).reshape(28,28,1)

   
def inverted(X):
  filling_val = [255, 127, 80, 180, 150]
  return np.array(abs((X) - np.full((X.shape),np.random.choice(filling_val)))).reshape(28,28,1)
  
patterns = []
def load_patterns():
  path = "patterns/"
  for i in range(3):
    im = pImage.open(f"{path}pat{i+1}.png")
    im_np = np.asarray(im).reshape((28,28,1))
    patterns.append(im_np)
load_patterns()
def pattern_add(X):
  i = np.random.choice(3)
  X = X + patterns[i]
  return X

def bounding_box(X):
  img = np.array(X, np.uint8)
  ret, thresh = cv2.threshold(img , 127, 255, cv2.THRESH_BINARY)
  # print(thresh.shape)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if(len(contours) < 3): # in order to not work with noisy image as it will be corrupted
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
    new_img = (img)[y:y+h, x:x+w]
    img = cv2.resize(new_img, (28,28))

  return img.reshape((28,28,1))

# bg_tot = []
# def load_bg():
#   path = "bg/"
#   for i in range(10):
#     im = pImage.open(f"{path}bg{i+1}.jpeg")
#     im_np = np.asarray(im)
#     # print(im_np.shape)
#     bg_tot.append(im_np)
# load_bg()

# def bg_mnist(X): 
#   p = np.random.choice(9)
#   img = np.array(X, np.uint8)
#   ret, thresh = cv2.threshold(img , 127, 255, cv2.THRESH_BINARY)
#   X = thresh.reshape((28,28,1))*255
#   bg_big = bg_tot[p+1]
#   x_coord, y_coord = np.random.randint(bg_big.shape[1]-28), np.random.randint(bg_big.shape[0]-28)
#   bg = bg_big[y_coord:y_coord+28, x_coord:x_coord+28]
#   for i in range(28):
#     for j in range(28):
#       if(X[i,j,0]  == 0): 
#         #print(bg[i,j])
#         X[i,j,0] = bg[i,j][0]
#       else:
#         if(p > 0.5):
#           X[i,j,0] = 255
#         else:
#           X[i,j, 0] = 0 
#   return X  


def imgaug_pre(X):
  sometimes = lambda aug: iaa.Sometimes(0.5, aug)
  # iaa.SomeOf((0, 5), [])

  seq = iaa.Sequential([
    sometimes(iaa.ElasticTransformation(alpha=(0, 2.5), sigma=2)),
    sometimes(iaa.Sharpen(alpha=(0, 0.5))), # sharpen images
    sometimes(iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.MotionBlur(k=3)
    ])),
    sometimes(iaa.AdditiveGaussianNoise(loc=1, scale=(0.0, 70), per_channel=0.5)),
  ])
  return seq(images=X.copy())


def preprocess_func(X):
  augs = [
          # bounding_box,0.7, #c: 0.5, 0.7, 0.8, x1
          # imgaug_pre,0.4,
          gaussian_blur,0.2, #c: 0.1, 0.2, x0.5
          dilation,0.2,#c: 0.1, 0.2, x0.3
          erosion,0.2,#c: 0.1, 0.2, x0.3
          gaussian_noise, 0.5,
          opening,0.2, #c: 0.1, 0.15, x0.2
          closing,0.2, #c: 0.1, 0.15, x0.2
          pattern_add, 0.5,
          inverted,0.4, #c: 0.1, 0.3, 0.4, x0.6
          ]
          
  num_augs = len(augs)
  p = np.random.random_sample((num_augs,))
  # print(p)

  for i in range(num_augs//2):
    thresh = augs[i*2+1]
    aug = augs[i*2]
    if(p[i] <= thresh):
      X = aug(X.copy())
  return X

generator = ImageDataGenerator(preprocessing_function=preprocess_func,
                              featurewise_center=False,  # set input mean to 0 over the dataset
                              samplewise_center=False,  # set each sample mean to 0
                              featurewise_std_normalization=False,  # divide inputs by std of the dataset
                              samplewise_std_normalization=False,  # divide each input by its std
                              zca_whitening=False,  # apply ZCA whitening
                              zca_epsilon=1e-06,    # epsilon for ZCA whitening. Default is 1e-6.
                              rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180) 30
                              zoom_range = 0.005,#0.1, # Randomly zoom image 
                              width_shift_range=0.1,#0.2,  # randomly shift images horizontally (fraction of total width) 0.1
                              height_shift_range=0.1,#0.3,  # randomly shift images vertically (fraction of total height) 0.1
                              horizontal_flip=False,  # randomly flip images
                              vertical_flip=False,
                              brightness_range=[1,1.5],#[30,80], # Tuple or list of two floats. Range for picking a brightness shift value from.
                              shear_range=0.3,
                              channel_shift_range=0.0,
                              fill_mode="nearest",
                              cval=0.0,
                              rescale=None,
                              data_format=None,
                              validation_split=0.15,)



# # iterator
# aug_iter = generator.flow(x_train, batch_size=1)

# nc = 10
# # generate samples and plot
# fig, ax = plt.subplots(nrows=1, ncols=nc, figsize=(15,15))

# # generate batch of images
# for i in range(nc):

# 	# convert to unsigned integers
# 	image = next(aug_iter)[0]
 
# 	# plot image
# 	ax[i].imshow(image.squeeze(), cmap='gray')
# 	ax[i].axis('off')
# plt.show()

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

lr_schedule = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=6)

# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logdir)



"""# 6. Compile the mode"""
model = make_model9()
model.compile(optimizer=Nadam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""## Plot the model"""
# print(model.summary())
# plot_model(model)

"""# 7. Training and testing"""
batch_size = 512 #c:512
num_epochs = 100

history = model.fit(generator.flow(x_train, y_train, batch_size=batch_size, seed=7),
          validation_data=generator.flow(x_train, y_train, batch_size=batch_size, subset="validation", seed=7),
          steps_per_epoch=len(x_train)/batch_size,
          epochs=num_epochs, 
          # verbose=2,
          # use_multiprocessing=True,
          # workers=4,
          callbacks=[
                     early_stopping,
                     lr_schedule])

print(model.evaluate(x_test, y_test))

"""# 8. History and saving the model"""
model.save("model.h5")

"""## Plots"""
# Plot the loss and accuracy curves for training and validation 
# fig, ax = plt.subplots(2,1, figsize=(18, 10))
# ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)
# plt.show()
"""## Confusion Matrix"""
# fig = plt.figure(figsize=(10, 10)) # Set Figure

# y_pred = model.predict(x_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels
# Y_test = y_test # Decode labels

# mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix

# # Plot Confusion matrix
# sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
# plt.xlabel('Predicted Values')
# plt.ylabel('True Values')
# plt.show()


# Tensorboard
"""
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# %tensorboard --logdir logs
"""
