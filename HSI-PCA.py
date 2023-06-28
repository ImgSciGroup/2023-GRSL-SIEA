from __future__ import print_function
import warnings

import cv2

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import random
import scipy.ndimage


import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils

from keras.models import Model
import numpy as np
from sklearn.svm import SVC

def loadIndianPinesData():
    """
    load data
    :return: data, labels
    """
    data_path = os.path.join(os.getcwd(), './paviau')
    data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']

    labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    print(data.shape)
    return data, labels

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=5, removeZeroLabels = True):
    """
    创建批数据
    :param X:
    :param y:
    :param windowSize:
    :param removeZeroLabels:
    :return:
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def applyPCA(X, numComponents):
    """
    使用PCA
    :param X: data
    :param numComponents: 要保留的组件数量
    :return: 应用PCA后的数据
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def splitTrainTestSet(X, y, testRatio=0.10):
    """
    分割数据集
    :param X:　
    :param y:
    :param testRatio: 分割因子
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)


    return X_train, X, y_train, y



if __name__ == '__main__':
    numComponents = 3
    windowSize = 1
    testRatio = 42000  #salinas53969

    # 加载数据

    Band = [5, 10, 15, 20, 25, 30, 40, 50]
    for B in range(len(Band)):
        X, y = loadIndianPinesData()
        # 使用PCA
        X,pca = applyPCA(X,numComponents=Band[B])
        XPatches, yPatches = createPatches(X, y, windowSize=windowSize)

        X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
        trainfeaturearray = []
        trainlabelarray = []
        for i in range(X_train.shape[0]):
            cube = []
            for j in range(X_train.shape[3]):
                cube.append(np.mean(X_train[i,:,:,j]))
            trainfeaturearray.append(cube)
            trainlabelarray.append(y_train[i])
        print(len(trainlabelarray))

        XPatches_test, yPatches_test = createPatches(X, y, windowSize=windowSize, removeZeroLabels=False)
        X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches_test, yPatches_test, testRatio)

        testfeaturearray = []
        for i in range(X_test.shape[0]):
            testcube = []
            for j in range(X_test.shape[3]):
                testcube.append(np.mean(X_test[i,:,:,j]))
            testfeaturearray.append(testcube)





        svm = SVC(C= 10, kernel='rbf',gamma='auto')
        svm.fit(trainfeaturearray, trainlabelarray)

        test_output = svm.predict(testfeaturearray)
        print(test_output.shape)
        testout = np.zeros((610, 340), np.uint8)
        index = 0
        for i in range(testout.shape[0]):
            for j in range(testout.shape[1]):
                testout[i,j] = test_output[index]+1
                index = index+1
        cv2.imwrite("./paviau_{}.bmp".format(Band[B]), testout)



#
# import numpy as np
# tmp = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# mean = np.mean(tmp)
# print(mean)