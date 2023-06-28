import cv2
import numpy as np
import os
# paviau colormodel
# imgpath = 'D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment5_Feature_test\Paviau\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (254, 0, 0)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 255, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 0, 255)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0, 139, 1)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (255, 255, 0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (254, 0, 255)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (9, 218, 249)
#                 if gt_img[i][j][0] == 8:
#                     resultimg[i][j] = (239, 32, 160)
#                 if gt_img[i][j][0] == 9:
#                     resultimg[i][j] = (96, 48, 176)
#         cv2.imwrite(imgpath+str('Paviau_color{}'.format(file)), resultimg)

#

#paviac color model

# imgpath = 'D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\paviac\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (255, 0, 0)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 130, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 255, 0)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0, 0, 255)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0, 90, 185)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (190, 190, 190)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (255, 255, 0)
#                 if gt_img[i][j][0] == 8:
#                     resultimg[i][j] = (86, 155, 255)
#                 if gt_img[i][j][0] == 9:
#                     resultimg[i][j] = (0, 255, 255)
#         cv2.imwrite(imgpath+str('Paviac_color{}'.format(file)), resultimg)



#ksc colormodel
# imgpath = 'D:\Firstwork_2022_9\AutomaticAreagrow\Experiment2\KSC\RF\RF\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (191, 0, 0)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (255, 0, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (255, 64, 0)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (255, 128, 0)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (255, 191, 0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (255, 255, 0)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (191, 255, 64)
#                 if gt_img[i][j][0] == 8:
#                     resultimg[i][j] = (128, 255, 128)
#                 if gt_img[i][j][0] == 9:
#                     resultimg[i][j] = (64, 255, 191)
#                 if gt_img[i][j][0] == 10:
#                     resultimg[i][j] = (0, 255, 255)
#                 if gt_img[i][j][0] == 11:
#                     resultimg[i][j] = (0, 191, 255)
#                 if gt_img[i][j][0] == 12:
#                     resultimg[i][j] = (0, 128, 255)
#                 if gt_img[i][j][0] == 13:
#                     resultimg[i][j] = (0, 64, 255)
#
#         cv2.imwrite(imgpath+str('KSC_color{}'.format(file)), resultimg)


#Salinas colormodel
imgpath = 'D:\Paper03-20230410-Active_learing_smote\Code\deeplearn\CNN_Enhanced_GCN-master\\'
files = os.listdir(imgpath)
for file in files:
    if file.endswith('.bmp'):
        path = imgpath+str('\\')+file
        gt_img = cv2.imread(path)
        resultimg = np.zeros(gt_img.shape, np.uint8)
        for i in range(gt_img.shape[0]):
            for j in range(gt_img.shape[1]):

                if gt_img[i][j][0] == 1:
                    resultimg[i][j] = (191, 0, 0)
                if gt_img[i][j][0] == 2:
                    resultimg[i][j] = (255, 0, 0)
                if gt_img[i][j][0] == 3:
                    resultimg[i][j] = (255, 64, 0)
                if gt_img[i][j][0] == 4:
                    resultimg[i][j] = (255, 128, 0)
                if gt_img[i][j][0] == 5:
                    resultimg[i][j] = (255, 191, 0)
                if gt_img[i][j][0] == 6:
                    resultimg[i][j] = (255, 255, 0)
                if gt_img[i][j][0] == 7:
                    resultimg[i][j] = (191, 255, 64)
                if gt_img[i][j][0] == 8:
                    resultimg[i][j] = (128, 255, 128)
                if gt_img[i][j][0] == 9:
                    resultimg[i][j] = (64, 255, 191)
                if gt_img[i][j][0] == 10:
                    resultimg[i][j] = (0, 255, 255)
                if gt_img[i][j][0] == 11:
                    resultimg[i][j] = (0, 191, 255)
                if gt_img[i][j][0] == 12:
                    resultimg[i][j] = (0, 128, 255)
                if gt_img[i][j][0] == 13:
                    resultimg[i][j] = (0, 64, 255)
                if gt_img[i][j][0] == 14:
                    resultimg[i][j] = (0, 0, 255)
                if gt_img[i][j][0] == 15:
                    resultimg[i][j] = (0, 0, 191)
                if gt_img[i][j][0] == 16:
                    resultimg[i][j] = (0, 0, 128)

        cv2.imwrite(imgpath+str('salinas_color{}'.format(file)), resultimg)




#WHU-Longkou
# imgpath = 'D:\Paper03-20230410-Active_learing_smote\Experiment2\WHU_Longkou\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 0:
#                     resultimg[i][j] = (0, 0, 0)
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (0, 0, 255)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 154, 238)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 255, 255)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0, 255, 0)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (255, 255, 0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (139, 139, 0)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (255, 0, 0)
#                 if gt_img[i][j][0] == 8:
#                     resultimg[i][j] = (255, 255, 255)
#                 if gt_img[i][j][0] == 9:
#                     resultimg[i][j] = (240, 32, 160)
#
#
#
#         cv2.imwrite(imgpath+str('WHU_Longkou_color{}'.format(file)), resultimg)



# #Scale32
# imgpath = 'D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\scale32\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (120,120,120)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 255, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 70, 152)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (60,60,60)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0,64,0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (177, 94, 54)
#
#         cv2.imwrite(imgpath+str('Scale32_color{}'.format(file)), resultimg)


#QB14
# imgpath = '..\..\Experiment1_Ourmethod\ZH6\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (100,100,100)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (36, 149, 232)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 125, 0)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0,255,0)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (150,0,0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (0, 80, 150)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (255, 150, 150)
#
#         cv2.imwrite(imgpath+str('ZH6_color{}'.format(file)), resultimg)



#ZH3
# imgpath = '..\..\Experiment1_Ourmethod\ZH3\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (36,149,232)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 255, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 255, 255)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0,80,150)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0,125,0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (150, 0, 0)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (100, 100, 100)
#
#         cv2.imwrite(imgpath+str('ZH3_color{}'.format(file)), resultimg)


#JX01
# imgpath = 'D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\jx01\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (128,255,255)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (0, 252, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (0, 64, 128)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (240,0,255)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0,64,0)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (245, 5, 0)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (0, 128, 128)
#
#         cv2.imwrite(imgpath+str('JX01_color{}'.format(file)), resultimg)




#GF-02
# imgpath = '..\..\Experiment1_Ourmethod\GF-2-SUB\SCI\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (200,0,0)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (150, 150, 250)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (200, 0, 200)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0,0,200)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0,250,150)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (250, 200, 0)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (0, 200, 0)
#
#         cv2.imwrite(imgpath+str('JX01_color{}'.format(file)), resultimg)

#GF2_PMS2__L1A0000718813-MSS2
# imgpath = '..\..\data\GF2WHU-image_RGB\\'
# files = os.listdir(imgpath)
# for file in files:
#     if file.endswith('.bmp'):
#         path = imgpath+str('\\')+file
#         gt_img = cv2.imread(path)
#         resultimg = np.zeros(gt_img.shape, np.uint8)
#         for i in range(gt_img.shape[0]):
#             for j in range(gt_img.shape[1]):
#                 if gt_img[i][j][0] == 1:
#                     resultimg[i][j] = (200,0,0)
#                 if gt_img[i][j][0] == 2:
#                     resultimg[i][j] = (200, 150, 0)
#                 if gt_img[i][j][0] == 3:
#                     resultimg[i][j] = (250, 200, 0)
#                 if gt_img[i][j][0] == 4:
#                     resultimg[i][j] = (0,200,0)
#                 if gt_img[i][j][0] == 5:
#                     resultimg[i][j] = (0,250,150)
#                 if gt_img[i][j][0] == 6:
#                     resultimg[i][j] = (200, 0, 200)
#                 if gt_img[i][j][0] == 7:
#                     resultimg[i][j] = (0, 0, 200)
#                 if gt_img[i][j][0] == 8:
#                     resultimg[i][j] = (150, 150, 200)
#                 if gt_img[i][j][0] == 9:
#                     resultimg[i][j] = (150, 150, 250)
#                 if gt_img[i][j][0] == 10:
#                     resultimg[i][j] = (150, 200, 150)
#
#
#         cv2.imwrite(imgpath+str('JX01_color{}'.format(file)), resultimg)


#单个
# gt_img = cv2.imread("D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\paviau\\paviau-ssfcn1.bmp")
# resultimg = np.zeros(gt_img.shape, np.uint8)
# for i in range(gt_img.shape[0]):
#     for j in range(gt_img.shape[1]):
#         if gt_img[i][j][0] == 1:
#             resultimg[i][j] = (254, 0, 0)
#         if gt_img[i][j][0] == 2:
#             resultimg[i][j] = (0, 255, 0)
#         if gt_img[i][j][0] == 3:
#             resultimg[i][j] = (0, 0, 255)
#         if gt_img[i][j][0] == 4:
#             resultimg[i][j] = (0, 139, 1)
#         if gt_img[i][j][0] == 5:
#             resultimg[i][j] = (255, 255, 0)
#         if gt_img[i][j][0] == 6:
#             resultimg[i][j] = (254, 0, 255)
#         if gt_img[i][j][0] == 7:
#             resultimg[i][j] = (9, 218, 249)
#         if gt_img[i][j][0] == 8:
#             resultimg[i][j] = (239, 32, 160)
#         if gt_img[i][j][0] == 9:
#             resultimg[i][j] = (96, 48, 176)
#
# cv2.imwrite("D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\paviau\paviau-ssfcn1.png", resultimg)
#
# gt_img = cv2.imread("D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\paviac\\paviac-ssrn.bmp")
# resultimg = np.zeros(gt_img.shape, np.uint8)
# for i in range(gt_img.shape[0]):
#     for j in range(gt_img.shape[1]):
#         if gt_img[i][j][0] == 1:
#             resultimg[i][j] = (254, 0, 0)
#         if gt_img[i][j][0] == 2:
#             resultimg[i][j] = (0, 139, 1)
#         if gt_img[i][j][0] == 3:
#             resultimg[i][j] = (255, 255, 0)
#         if gt_img[i][j][0] == 4:
#             resultimg[i][j] = (239, 32, 160)
#         if gt_img[i][j][0] == 5:
#             resultimg[i][j] = (9, 218, 249)
#         if gt_img[i][j][0] == 6:
#             resultimg[i][j] = (0, 0, 255)
#         if gt_img[i][j][0] == 7:
#             resultimg[i][j] = (96, 48, 176)
#         if gt_img[i][j][0] == 8:
#             resultimg[i][j] = (0, 255, 0)
#         if gt_img[i][j][0] == 9:
#             resultimg[i][j] = (254, 0, 255)
#
# cv2.imwrite("D:\Paper02_20221101-SpaceAnd_spectrum_Classification\Experiment8-deeplearning\paviac\\paviac-ssrn.png", resultimg)

#ksccolormodel
# gt_img = cv2.imread("D:\Firstwork_2022_9\DeepLearning\KSC\DeepHypex\\resultgroungtruth3.bmp")
# resultimg = np.zeros(gt_img.shape, np.uint8)
# for i in range(gt_img.shape[0]):
#     for j in range(gt_img.shape[1]):
#         if gt_img[i][j][0] == 1:
#             resultimg[i][j] = (191, 0, 0)
#         if gt_img[i][j][0] == 2:
#             resultimg[i][j] = (255, 0, 0)
#         if gt_img[i][j][0] == 3:
#             resultimg[i][j] = (255, 64, 0)
#         if gt_img[i][j][0] == 4:
#             resultimg[i][j] = (255, 128, 0)
#         if gt_img[i][j][0] == 5:
#             resultimg[i][j] = (255, 191, 0)
#         if gt_img[i][j][0] == 6:
#             resultimg[i][j] = (255, 255, 0)
#         if gt_img[i][j][0] == 7:
#             resultimg[i][j] = (191, 255, 64)
#         if gt_img[i][j][0] == 8:
#             resultimg[i][j] = (128, 255, 128)
#         if gt_img[i][j][0] == 9:
#             resultimg[i][j] = (64, 255, 191)
#         if gt_img[i][j][0] == 10:
#             resultimg[i][j] = (0, 255, 255)
#         if gt_img[i][j][0] == 11:
#             resultimg[i][j] = (0, 191, 255)
#         if gt_img[i][j][0] == 12:
#             resultimg[i][j] = (0, 128, 255)
#         if gt_img[i][j][0] == 13:
#             resultimg[i][j] = (0, 64, 255)
#
#
# cv2.imwrite("D:\Firstwork_2022_9\DeepLearning\KSC\DeepHypex\\result_color.png", resultimg)


# salinscolormodel
# gt_img = cv2.imread("D:\Firstwork_2022_9\DeepLearning\Salinas\A2S2resnet\\result_gt.bmp")
# resultimg = np.zeros(gt_img.shape, np.uint8)
# for i in range(gt_img.shape[0]):
#     for j in range(gt_img.shape[1]):
#         if gt_img[i][j][0] == 1:
#             resultimg[i][j] = (191, 0, 0)
#         if gt_img[i][j][0] == 2:
#             resultimg[i][j] = (255, 0, 0)
#         if gt_img[i][j][0] == 3:
#             resultimg[i][j] = (255, 64, 0)
#         if gt_img[i][j][0] == 4:
#             resultimg[i][j] = (255, 128, 0)
#         if gt_img[i][j][0] == 5:
#             resultimg[i][j] = (255, 191, 0)
#         if gt_img[i][j][0] == 6:
#             resultimg[i][j] = (255, 255, 0)
#         if gt_img[i][j][0] == 7:
#             resultimg[i][j] = (191, 255, 64)
#         if gt_img[i][j][0] == 8:
#             resultimg[i][j] = (128, 255, 128)
#         if gt_img[i][j][0] == 9:
#             resultimg[i][j] = (64, 255, 191)
#         if gt_img[i][j][0] == 10:
#             resultimg[i][j] = (0, 255, 255)
#         if gt_img[i][j][0] == 11:
#             resultimg[i][j] = (0, 191, 255)
#         if gt_img[i][j][0] == 12:
#             resultimg[i][j] = (0, 128, 255)
#         if gt_img[i][j][0] == 13:
#             resultimg[i][j] = (0, 64, 255)
#         if gt_img[i][j][0] == 14:
#             resultimg[i][j] = (0, 0, 255)
#         if gt_img[i][j][0] == 15:
#             resultimg[i][j] = (0, 0, 191)
#         if gt_img[i][j][0] == 16:
#             resultimg[i][j] = (0, 0, 128)
#
#
#
# cv2.imwrite("D:\Firstwork_2022_9\DeepLearning\Salinas\A2S2resnet\\result_color1.png", resultimg)