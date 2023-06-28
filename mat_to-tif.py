import os

import cv2
import scipy.io as sio
import numpy as np
import random

def loadData():
    """
    加载数据
    :return: data, labels
    """
    data_path = os.path.join(os.getcwd(), './Data/WHU_Longkou')
    data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
    labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    return data, labels


if __name__=='__main__':

    X, y = loadData()
    h,w = y.shape[:2]
    gt = np.zeros((h,w), np.uint8)
    for i in range(h):
        for j in range(w):
            gt[i,j] = y[i,j]
    cv2.imwrite("./Data/WHU_Longkou/WHU_Longkou_gt.bmp", gt)
