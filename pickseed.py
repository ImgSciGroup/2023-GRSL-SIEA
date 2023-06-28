import os
import scipy.io as sio
import numpy as np
import random
# def loadData():
#     """
#     加载数据
#     :return: data, labels
#     """
#     data_path = os.path.join(os.getcwd(), './Data/WHU_Longkou')
#     data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
#     labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
#     return data, labels
#
#
#
# if __name__=='__main__':
#
#     X, y = loadData()
#     classnum = len(np.unique(y)) - 1
#     M1_data1,M1_data2,M1_data3,M1_data4,M1_data5,M1_data6,M1_data7,M1_data8,M1_data9 = [[] for x in range(classnum)]
#     M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7, M1_train8, M1_train9= [[] for x in range(classnum)]
#     for row in range(y.shape[0]):
#         for col in range(y.shape[1]):
#             if y[row, col] == 1:
#                 M1_data1.append((row, col))
#             if y[row, col] == 2:
#                 M1_data2.append((row, col))
#             if y[row, col] == 3:
#                 M1_data3.append((row, col))
#             if y[row, col] == 4:
#                 M1_data4.append((row, col))
#             if y[row, col] == 5:
#                 M1_data5.append((row, col))
#             if y[row, col] == 6:
#                 M1_data6.append((row, col))
#             if y[row, col] == 7:
#                 M1_data7.append((row, col))
#             if y[row, col] == 8:
#                 M1_data8.append((row, col))
#             if y[row, col] == 9:
#                 M1_data9.append((row, col))
#
#     trainsize = 10
#
#     for i in range(classnum):
#         initseedcoordinate = []#收集标注在图上
#         for j in range(trainsize):
#             length = len(locals()['M1_data{0}'.format(i + 1)])
#             seedindex = random.randint(0, length-1)
#             seed = locals()['M1_data{0}'.format(i+1)].pop(seedindex)
#             locals()['M1_train{0}'.format(i+1)].append(seed)
#     with open('../../Experiment1/WHU_Longkou\\pickseed\\train0.txt', 'w') as train:
#         for i in range(classnum):
#             for j in range(len(locals()['M1_train{0}'.format(i + 1)])):
#                 write_seed = locals()['M1_train{0}'.format(i + 1)][j]
#
#                 train.write('{0}'.format(i + 1) + "," + '{0}'.format(write_seed) + '\n')
#     train.close()
#
#
#     for m in range(21):
#         trainsize = 11
#         for i in range(classnum):
#             initseedcoordinate = []#收集标注在图上
#             for j in range(trainsize):
#                 length = len(locals()['M1_data{0}'.format(i + 1)])
#                 seedindex = random.randint(0, length-1)
#                 seed = locals()['M1_data{0}'.format(i+1)].pop(seedindex)
#                 locals()['M1_train{0}'.format(i+1)].append(seed)
#         with open('../../Experiment1/WHU_Longkou\\pickseed\\train{0}.txt'.format(m+1), 'w') as train:
#             for i in range(classnum):
#                 for j in range(len(locals()['M1_train{0}'.format(i + 1)])):
#                     write_seed = locals()['M1_train{0}'.format(i + 1)][j]
#
#                     train.write('{0}'.format(i + 1) + "," + '{0}'.format(write_seed) + '\n')
#         train.close()
#





#WHU-LONGKOU


def loadData():
    """
    加载数据
    :return: data, labels
    """
    data_path = os.path.join(os.getcwd(), './Data/Salinas')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    return data, labels



if __name__=='__main__':

    X, y = loadData()
    classnum = len(np.unique(y)) - 1
    M1_data1, M1_data2, M1_data3, M1_data4, M1_data5, M1_data6, M1_data7, M1_data8, M1_data9, M1_data10, M1_data11, M1_data12, M1_data13, M1_data14, M1_data15, M1_data16 = [
        [] for x in range(classnum)]
    M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7, M1_train8, M1_train9, M1_train10, M1_train11, M1_train12, M1_train13, M1_train14, M1_train15, M1_train16 = [
        [] for x in range(classnum)]

    for row in range(y.shape[0]):
        for col in range(y.shape[1]):
            if y[row, col] == 1:
                M1_data1.append((row, col))
            if y[row, col] == 2:
                M1_data2.append((row, col))
            if y[row, col] == 3:
                M1_data3.append((row, col))
            if y[row, col] == 4:
                M1_data4.append((row, col))
            if y[row, col] == 5:
                M1_data5.append((row, col))
            if y[row, col] == 6:
                M1_data6.append((row, col))
            if y[row, col] == 7:
                M1_data7.append((row, col))
            if y[row, col] == 8:
                M1_data8.append((row, col))
            if y[row, col] == 9:
                M1_data9.append((row, col))
            if y[row, col] == 10:
                M1_data10.append((row, col))
            if y[row, col] == 11:
                M1_data11.append((row, col))
            if y[row, col] == 12:
                M1_data12.append((row, col))
            if y[row, col] == 13:
                M1_data13.append((row, col))
            if y[row, col] == 14:
                M1_data14.append((row, col))
            if y[row, col] == 15:
                M1_data15.append((row, col))
            if y[row, col] == 16:
                M1_data16.append((row, col))

    trainsize = 10

    for i in range(classnum):
        initseedcoordinate = []#收集标注在图上
        for j in range(trainsize):
            length = len(locals()['M1_data{0}'.format(i + 1)])
            seedindex = random.randint(0, length-1)
            seed = locals()['M1_data{0}'.format(i+1)].pop(seedindex)
            locals()['M1_train{0}'.format(i+1)].append(seed)
    with open('../../Experiment1/Salinas\\train0.txt', 'w') as train:
        for i in range(classnum):
            for j in range(len(locals()['M1_train{0}'.format(i + 1)])):
                write_seed = locals()['M1_train{0}'.format(i + 1)][j]

                train.write('{0}'.format(i + 1) + "," + '{0}'.format(write_seed) + '\n')
    train.close()


    for m in range(21):
        trainsize = 11
        for i in range(classnum):
            initseedcoordinate = []#draw in data
            for j in range(trainsize):
                length = len(locals()['M1_data{0}'.format(i + 1)])
                seedindex = random.randint(0, length-1)
                seed = locals()['M1_data{0}'.format(i+1)].pop(seedindex)
                locals()['M1_train{0}'.format(i+1)].append(seed)
        with open('../../Experiment1/Salinas\\train{0}.txt'.format(m+1), 'w') as train:
            for i in range(classnum):
                for j in range(len(locals()['M1_train{0}'.format(i + 1)])):
                    write_seed = locals()['M1_train{0}'.format(i + 1)][j]

                    train.write('{0}'.format(i + 1) + "," + '{0}'.format(write_seed) + '\n')
        train.close()

