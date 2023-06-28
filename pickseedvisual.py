import glob
# import os
# import cv2
# import linecache
# seedpath = 'D:\Paper03-20230410-Active_learing_smote\Experiment1\Salinas\pickseed\\'#根路径
# files = os.listdir(seedpath)
#
# for k in range(0, len(files)):
#     image = cv2.imread('D:\Paper03-20230410-Active_learing_smote\Data\Salinas/Salinas.tif')
#     with open(seedpath + 'train{}.txt'.format(k), 'r') as pickseed:
#         for j in range(1, len(open(seedpath + 'train{}.txt'.format(k)).readlines()) + 1):
#             lindata = linecache.getline(seedpath + 'train{}.txt'.format(k), j)
#             id = int(lindata.split(',')[0])
#             coordx = int(lindata.split(',')[1].split('(')[1])
#             coordy = int(lindata.split(' ')[1].split(')')[0])
#             if id == 1:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 0, 255), -1)
#             if id == 2:
#                 cv2.circle(image, (coordy, coordx), 1, (255, 0, 0), -1)
#             if id == 3:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 255, 0), -1)
#             if id == 4:
#                 cv2.circle(image, (coordy, coordx), 1, (255, 0, 255), -1)
#             if id == 5:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 255, 255), -1)
#             if id == 6:
#                 cv2.circle(image, (coordy, coordx), 1, (128, 0, 128), -1)
#             if id == 7:
#                 cv2.circle(image, (coordy, coordx), 1, (255, 255, 255), -1)
#             if id == 8:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 0, 0), -1)
#             if id == 9:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 128, 255), -1)
#             if id == 10:
#                 cv2.circle(image, (coordy, coordx), 1, (128, 0, 255), -1)
#             if id == 11:
#                 cv2.circle(image, (coordy, coordx), 1, (255, 0, 128), -1)
#             if id == 12:
#                 cv2.circle(image, (coordy, coordx), 1, (255, 128, 0), -1)
#             if id == 13:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 255, 128), -1)
#             if id == 14:
#                 cv2.circle(image, (coordy, coordx), 1, (128, 255, 0), -1)
#             if id == 15:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 0, 128), -1)
#             if id == 16:
#                 cv2.circle(image, (coordy, coordx), 1, (0, 128, 0), -1)
#                 # cv2.imshow("img",image)
#     cv2.imwrite("D:\Paper03-20230410-Active_learing_smote\Experiment1\Salinas\pickseed/pickseedSalinas{0}.bmp".format(k), image)



import glob
import os
import cv2
import linecache
seedpath = 'D:\Paper03-20230410-Active_learing_smote\Experiment1\WHU_Longkou\pickseed\\'#根root path
files = os.listdir(seedpath)

for k in range(0, len(files)):
    image = cv2.imread('D:\Paper03-20230410-Active_learing_smote\Data\WHU_Longkou/WHU-Hi-LongKou(97,70,30).tif')
    with open(seedpath + 'train{}.txt'.format(k), 'r') as pickseed:
        for j in range(1, len(open(seedpath + 'train{}.txt'.format(k)).readlines()) + 1):
            lindata = linecache.getline(seedpath + 'train{}.txt'.format(k), j)
            id = int(lindata.split(',')[0])
            coordx = int(lindata.split(',')[1].split('(')[1])
            coordy = int(lindata.split(' ')[1].split(')')[0])
            if id == 1:
                cv2.circle(image, (coordy, coordx), 1, (255, 255, 255), -1)
            if id == 2:
                cv2.circle(image, (coordy, coordx), 1, (255, 0, 0), -1)
            if id == 3:
                cv2.circle(image, (coordy, coordx), 1, (0, 255, 0), -1)
            if id == 4:
                cv2.circle(image, (coordy, coordx), 1, (255, 0, 255), -1)
            if id == 5:
                cv2.circle(image, (coordy, coordx), 1, (0, 255, 255), -1)
            if id == 6:
                cv2.circle(image, (coordy, coordx), 1, (128, 255, 128), -1)
            if id == 7:
                cv2.circle(image, (coordy, coordx), 1, (0, 0, 255), -1)
            if id == 8:
                cv2.circle(image, (coordy, coordx), 1, (0, 0, 0), -1)
            if id == 9:
                cv2.circle(image, (coordy, coordx), 1, (0, 128, 255), -1)

                # cv2.imshow("img",image)
    cv2.imwrite("D:\Paper03-20230410-Active_learing_smote\Experiment1\WHU_Longkou\pickseed/pickseedWHU{0}.bmp".format(k), image)

