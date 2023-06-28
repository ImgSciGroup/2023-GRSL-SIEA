from __future__ import print_function
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import scipy.io as sio
import os
import numpy as np
from sklearn.svm import SVC
import cv2
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc, cohen_kappa_score
import math
from collections import Counter
def loadData():
    """
    Load data
    :return: data, labels
    """
    data_path = os.path.join(os.getcwd(), './Data/Salinas')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    return data, labels

def applyPCA(X, numComponents):
    """
    使用PCA
    :param X: data
    :param numComponents: 要保留的组件数量
    :return: after applay pca data
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


def Outputlabel_probaility(Train_data1, Train_data2, Train_data3, Train_data4, Train_data5, Train_data6, Train_data7, Train_data8, Train_data9, Train_data10, Train_data11, Train_data12, Train_data13, Train_data14, Train_data15, Train_data16, M1_test):
    feature = []
    label = []
    for i in range(1, 17):
        for j in range(len(locals()['Train_data{0}'.format(i)])):
            feature.append(np.asarray(locals()['Train_data{0}'.format(i)][j].data[0]))
            label.append(int(locals()['Train_data{0}'.format(i)][j].Label))
    feature = np.asarray(feature)
    label = np.asarray(label)

    svc = SVC(C=10, degree=3, kernel='rbf', gamma=0.001, probability=True)

    svc.fit(feature, label)

    M1_truelabel = []

    M1_testdata=[]
    M1_testcoordinate = []
    for i in range(0, 16):
        teststruct = M1_test[i]
        for j in range(len(teststruct)):
            M1_truelabel.append(np.asarray(int(teststruct[j].Label)))

            M1_testdata.append(np.asarray(teststruct[j].data[0]))
            M1_testcoordinate.append(np.asarray(teststruct[j].coordinate[0]))

    M1_testdata = np.asarray(M1_testdata)

    M1_truelabel = np.asarray(M1_truelabel)

    M1_testcoordinate = np.asarray(M1_testcoordinate)
    predictlabel = []
    probabilitylabel = []
    for i in range(len(M1_testdata)):
        predict = svc.predict([M1_testdata[i]])
        probability = svc.predict_proba([M1_testdata[i]])
        predictlabel.append(predict)
        probabilitylabel.append(probability)
    predictlabel = np.asarray(predictlabel)
    predictlabel = list(np.array(predictlabel).flatten())
    probabilitylabel = np.asarray(probabilitylabel)
    probabilitylabel = np.squeeze(probabilitylabel)

    acccuracy = float(accuracy_score(M1_truelabel, predictlabel))
    print(acccuracy)

    return predictlabel,M1_testdata, M1_testcoordinate,probabilitylabel

def Calcuate_uncertain(M1_candidatepro1, M1_candidatepro2, M1_candidatepro3, M1_candidatepro4, M1_candidatepro5, M1_candidatepro6, M1_candidatepro7, M1_candidatepro8, M1_candidatepro9, M1_candidatepro10, M1_candidatepro11,M1_candidatepro12, M1_candidatepro13, M1_candidatepro14, M1_candidatepro15, M1_candidatepro16, classnum):
    M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16= [[] for x in range(classnum)]
    for i in range(classnum):
        for j in range(len(locals()['M1_candidatepro{0}'.format(i + 1)])):
            pro = locals()['M1_candidatepro{0}'.format(i + 1)][j]
            pro.sort()

            locals()['M1_prodiffer{0}'.format(i + 1)].append(pro[len(pro) - 1] - pro[len(pro) - 2])
    return M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16

def Calcuate_information(M1_candidatedata1, M1_candidatedata2, M1_candidatedata3, M1_candidatedata4, M1_candidatedata5, M1_candidatedata6, M1_candidatedata7,M1_candidatedata8, M1_candidatedata9, M1_candidatedata10, M1_candidatedata11, M1_candidatedata12, M1_candidatedata13, M1_candidatedata14,M1_candidatedata15, M1_candidatedata16, classnum):
    M1_infodata1, M1_infodata2, M1_infodata3, M1_infodata4, M1_infodata5, M1_infodata6, M1_infodata7, M1_infodata8, M1_infodata9, M1_infodata10, M1_infodata11, M1_infodata12, M1_infodata13, M1_infodata14, M1_infodata15, M1_infodata16 = [[] for x in range(classnum)]

    for i in range(classnum):
        for j in range(len(locals()['M1_candidatedata{0}'.format(i+1)])):
            mid_differ = []
            for k in range(len(locals()['M1_candidatedata{0}'.format(i+1)])):

                differ = abs(locals()['M1_candidatedata{0}'.format(i + 1)][j] - locals()['M1_candidatedata{0}'.format(i + 1)][k])
                mid_differ.append(differ)
            mid_differ = np.asarray(mid_differ)
            locals()['M1_infodata{0}'.format(i + 1)].append(np.sum(mid_differ))




    return M1_infodata1,M1_infodata2,M1_infodata3,M1_infodata4,M1_infodata5,M1_infodata6,M1_infodata7,M1_infodata8,M1_infodata9,M1_infodata10,M1_infodata11,M1_infodata12,M1_infodata13,M1_infodata14,M1_infodata15,M1_infodata16


def Normalization(M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16, classnum):
    for i in range(classnum):
        mindata = min(locals()['M1_prodiffer{0}'.format(i+1)])
        maxdata = max(locals()['M1_prodiffer{0}'.format(i+1)])
        for j in range(len(locals()['M1_prodiffer{0}'.format(i+1)])):
            normalizedata = (locals()['M1_prodiffer{0}'.format(i+1)][j]-mindata)/(maxdata-mindata)
            locals()['M1_prodiffer{0}'.format(i + 1)][j] = normalizedata
    return M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16



def Calcuate_score(M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16,M1_infodata1,M1_infodata2,M1_infodata3,M1_infodata4,M1_infodata5,M1_infodata6,M1_infodata7,M1_infodata8,M1_infodata9,M1_infodata10,M1_infodata11,M1_infodata12,M1_infodata13,M1_infodata14,M1_infodata15,M1_infodata16,classnum):
    B = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    score1, score2, score3, score4, score5, score6, score7, score8, score9, score10, score11, score12, score13, score14, score15, score16 = [
        [] for x in range(classnum)]

    for i in range(classnum):
        for b in range(len(B)):
            Score = []
            for j in range(len(locals()['M1_infodata{0}'.format(i + 1)])):
                score = locals()['M1_infodata{0}'.format(i + 1)][j] ** (B[b]) + \
                        locals()['M1_prodiffer{0}'.format(i + 1)][j] ** (1 - B[b])
                Score.append(score)

            locals()['score{0}'.format(i + 1)].append(Score)
    return score1, score2, score3, score4, score5, score6, score7, score8, score9, score10, score11, score12, score13, score14, score15, score16



def Pickseed(Score, Candidatecoordinate, Traindata, labelid, classnum):#Score has nine column scores, Candidatecoordinate candidate sample coordinates, Traindata: training sample structure
    traincoord = []#Put the training sample coordinates
    trainlabel = []#Put the training sample label

    for k in range(len(Traindata)):#First collect the training sample labels and coordinates
        label = int(Traindata[k].Label)
        coord = Traindata[k].coordinate[0]
        traincoord.append(coord)
        trainlabel.append(label)


    minindex = []
    for i in range(12):
        score = Score[i]#score
        A = []#index
        for j in range(len(score)):
            A.append(j)
        Z = zip(score, A)
        Z = sorted(Z)
        score, A = zip(*Z)#The scores are sorted from smallest to largest, and the index corresponds to them

        for l in range(len(A)):
            averagetraincoord = []
            traindistance = []  # Put the training sample coordinates away from the candidate sample distance
            coord = Candidatecoordinate[A[l]]
            for m in range(len(traincoord)):
                a = np.array(traincoord[m])
                a_x = a[0]
                a_y = a[1]

                b = np.array(coord)
                b_x = b[0]
                b_y = b[1]

                dis = math.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
                traindistance.append(dis)
            for t in range(classnum):
                sum = 0
                count = 0
                for s in range(len(trainlabel)):

                    if trainlabel[s] == t+1:
                        count = count+1
                        sum = sum+traindistance[s]
                averagetraincoord.append(sum/count)

            if i == 0:
                if averagetraincoord.index(min(averagetraincoord))+1 == labelid:
                    minindex.append(A[l])
                    break
                else:
                    continue
            else:
                if averagetraincoord.index(min(averagetraincoord))+1 == labelid and (A[l] not in minindex):
                    minindex.append(A[l])
                    break
                else:
                    continue
    return minindex



# def Label_spectral(pixel, Traindata, labelid):#Pixel high-quality samples, Traindata labeled samples, labelid: labeled sample labels
#     Classnum = np.unique(labelid)
#     minusarray = []#Put the distance between the pix and the mean distance of each type of sample in Traindata
#     for i in range(Classnum):
#         trainsample = []
#         for j in range(len(labelid)):
#             if labelid[j]==i+1:
#                 trainsample.append(Traindata[j])
#         mean = np.mean(trainsample)
#         std = np.std(trainsample)
#         if pixel<=mean+3*std and pixel>=mean-3*std:#The satisfaction is within the range of a certain type of sample distribution and is closest to the mean
#             minusarray.append(math.fabs(pixel-mean))
#         else:
#             minusarray.append(9999999)
#     spectral_label =minusarray.index(min(minusarray))
#     return spectral_label
#



# def Label_spatial(pixel_coord, Traindatacoordinate, labelid):#pixel_coord high-quality sample coordinates, Traindatacoordinate has labeled the sample coordinates, labelid: training sample label
#     num_k = []#Put the distance pixel_coord label for the last 6 known samples
#     distancearray = []
#     for i in range(Traindatacoordinate):
#         a_x = pixel_coord[0]
#         a_y = pixel_coord[1]
#         b_x = Traindatacoordinate[i][0]
#         b_y = Traindatacoordinate[i][1]
#         dis = math.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
#         distancearray.append(dis)
#     k = 6
#     while(k!=0):
#         minindex = distancearray.index(min(distancearray))
#         Traindatacoordinate.pop(minindex)
#         num_k.append(labelid.pop(minindex))
#         k=k-1
#     most_common = Counter(num_k)#统计num_k比如标签1出现3次，标签2出现1次等等
#     if most_common[0][1]>=k/2:
#         return most_common[0][0]
#     else:
#         return 9999999



def Drawimage(image, seedcoordinate, id):

    if id==1:
        for i in range(len(seedcoordinate)):
            # print(seedcoordinate[i])
            cv2.circle(image, (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 0, 255), -1)
    if id==2:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 0, 0), -1)
    if id==3:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 255, 0), -1)
    if id==4:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 0, 255), -1)
    if id==5:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 255, 255), -1)
    if id==6:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 255, 0), -1)
    if id==7:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 255, 255), -1)
    if id==8:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 0, 0), -1)
    if id==9:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 128, 255), -1)
    if id==10:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (128, 0, 255), -1)
    if id==11:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 0, 128), -1)
    if id==12:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (255, 128, 0), -1)
    if id==13:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 255, 128), -1)
    if id==14:
        for i in range(len(seedcoordinate)):
            cv2.circle(image, (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (128, 255, 0), -1)
    if id==15:
        for i in range(len(seedcoordinate)):
            cv2.circle(image, (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 0, 128), -1)
    if id==16:
        for i in range(len(seedcoordinate)):
            cv2.circle(image,  (seedcoordinate[i][0][1],seedcoordinate[i][0][0]), 1, (0, 128, 0), -1)
    cv2.imwrite("../../Data/Salinas/pivkseedSalinas{0}.tif".format(id), image)




def Combation(M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7,M1_train8, M1_train9, M1_train10, M1_train11, M1_train12, M1_train13, M1_train14,M1_train15, M1_train16, classnum):
    combation_train = []
    for i in range(classnum):
        for j in range(len(locals()['M1_train{0}'.format(i+1)])):
            combation_train.append(locals()['M1_train{}'.format(i+1)][j])
    return combation_train



def Labelclass(predictlabel,M1_testdata, M1_testcoordinate,probabilitylabel):#将分类结果标签进行分类，进行候选样本选取

    classnum = len(np.unique(predictlabel))
    M1_candidatedata1, M1_candidatedata2, M1_candidatedata3, M1_candidatedata4, M1_candidatedata5, M1_candidatedata6, M1_candidatedata7,M1_candidatedata8, M1_candidatedata9, M1_candidatedata10, M1_candidatedata11, M1_candidatedata12, M1_candidatedata13, M1_candidatedata14,M1_candidatedata15, M1_candidatedata16 = [[] for x in range(classnum)]
    M1_candidatecoordinate1, M1_candidatecoordinate2, M1_candidatecoordinate3, M1_candidatecoordinate4, M1_candidatecoordinate5, M1_candidatecoordinate6, M1_candidatecoordinate7, M1_candidatecoordinate8, M1_candidatecoordinate9, M1_candidatecoordinate10, M1_candidatecoordinate11, M1_candidatecoordinate12, M1_candidatecoordinate13, M1_candidatecoordinate14, M1_candidatecoordinate15, M1_candidatecoordinate16 = [[] for x in range(classnum)]
    M1_candidatepro1, M1_candidatepro2, M1_candidatepro3, M1_candidatepro4, M1_candidatepro5, M1_candidatepro6, M1_candidatepro7, M1_candidatepro8, M1_candidatepro9, M1_candidatepro10, M1_candidatepro11, M1_candidatepro12, M1_candidatepro13, M1_candidatepro14, M1_candidatepro15, M1_candidatepro16 = [[] for x in range(classnum)]
    for i in range(len(predictlabel)):
        if predictlabel[i]==1:
            M1_candidatecoordinate1.append(np.asarray(M1_testcoordinate[i]))
            M1_candidatedata1.append(M1_testdata[i])
            M1_candidatepro1.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==2:
            M1_candidatecoordinate2.append(M1_testcoordinate[i])
            M1_candidatedata2.append(M1_testdata[i])
            M1_candidatepro2.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==3:
            M1_candidatecoordinate3.append(M1_testcoordinate[i])
            M1_candidatedata3.append(M1_testdata[i])
            M1_candidatepro3.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==4:
            M1_candidatecoordinate4.append(M1_testcoordinate[i])
            M1_candidatedata4.append(M1_testdata[i])
            M1_candidatepro4.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==5:
            M1_candidatecoordinate5.append(M1_testcoordinate[i])
            M1_candidatedata5.append(M1_testdata[i])
            M1_candidatepro5.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==6:
            M1_candidatecoordinate6.append(M1_testcoordinate[i])
            M1_candidatedata6.append(M1_testdata[i])
            M1_candidatepro6.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==7:
            M1_candidatecoordinate7.append(M1_testcoordinate[i])
            M1_candidatedata7.append(M1_testdata[i])
            M1_candidatepro7.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==8:
            M1_candidatecoordinate8.append(M1_testcoordinate[i])
            M1_candidatedata8.append(M1_testdata[i])
            M1_candidatepro8.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==9:
            M1_candidatecoordinate9.append(M1_testcoordinate[i])
            M1_candidatedata9.append(M1_testdata[i])
            M1_candidatepro9.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==10:
            M1_candidatecoordinate10.append(M1_testcoordinate[i])
            M1_candidatedata10.append(M1_testdata[i])
            M1_candidatepro10.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==11:
            M1_candidatecoordinate11.append(M1_testcoordinate[i])
            M1_candidatedata11.append(M1_testdata[i])
            M1_candidatepro11.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==12:
            M1_candidatecoordinate12.append(M1_testcoordinate[i])
            M1_candidatedata12.append(M1_testdata[i])
            M1_candidatepro12.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==13:
            M1_candidatecoordinate13.append(M1_testcoordinate[i])
            M1_candidatedata13.append(M1_testdata[i])
            M1_candidatepro13.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==14:
            M1_candidatecoordinate14.append(M1_testcoordinate[i])
            M1_candidatedata14.append(M1_testdata[i])
            M1_candidatepro14.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==15:
            M1_candidatecoordinate15.append(M1_testcoordinate[i])
            M1_candidatedata15.append(M1_testdata[i])
            M1_candidatepro15.append(probabilitylabel[i])

    for i in range(len(predictlabel)):
        if predictlabel[i]==16:
            M1_candidatecoordinate16.append(M1_testcoordinate[i])
            M1_candidatedata16.append(M1_testdata[i])
            M1_candidatepro16.append(probabilitylabel[i])

    return M1_candidatedata1, M1_candidatedata2, M1_candidatedata3, M1_candidatedata4, M1_candidatedata5, M1_candidatedata6, M1_candidatedata7,M1_candidatedata8, M1_candidatedata9, M1_candidatedata10, M1_candidatedata11, M1_candidatedata12, M1_candidatedata13, M1_candidatedata14,M1_candidatedata15, M1_candidatedata16,M1_candidatecoordinate1, M1_candidatecoordinate2, M1_candidatecoordinate3, M1_candidatecoordinate4, M1_candidatecoordinate5, M1_candidatecoordinate6, M1_candidatecoordinate7, M1_candidatecoordinate8, M1_candidatecoordinate9, M1_candidatecoordinate10, M1_candidatecoordinate11, M1_candidatecoordinate12, M1_candidatecoordinate13, M1_candidatecoordinate14, M1_candidatecoordinate15, M1_candidatecoordinate16,M1_candidatepro1, M1_candidatepro2, M1_candidatepro3, M1_candidatepro4, M1_candidatepro5, M1_candidatepro6, M1_candidatepro7, M1_candidatepro8, M1_candidatepro9, M1_candidatepro10, M1_candidatepro11, M1_candidatepro12, M1_candidatepro13, M1_candidatepro14, M1_candidatepro15, M1_candidatepro16


def Remove(M1_test, All_pickseedcoordinate, classnum):
    # for i in range(classnum):
    #     for j in range(i*12,(i+1)*12):
    #         pickseed = All_pickseedcoordinate[i*12+j]
    #         # struct = M1_test[i]
    #         struct_length = len(M1_test[i])
    #         for k in range(struct_length):
    #             if pickseed == M1_test[i][k].coordinate:
    #                 del M1_test[i][k]
    #                 break
    # return M1_test

    for i in range(len(All_pickseedcoordinate)):

        for j in range(len(M1_test)):
            struct = M1_test[j]
            len_struct = len(struct)
            for k in range(len_struct):
                coordinate = M1_test[j][k].coordinate
                if coordinate == All_pickseedcoordinate[i]:
                    del M1_test[j][k]
                    break

    return M1_test



if __name__ == '__main__':
    X, y = loadData()
    X, pca = applyPCA(X, numComponents=10)
    classnum = len(np.unique(y))-1
    image = cv2.imread("../../Data/Salinas/Salinas.tif")
    #Define the training set, the test set, and all data sets, in the form [label, coordinate, data]
    M1_data1,M1_data2,M1_data3,M1_data4,M1_data5,M1_data6,M1_data7,M1_data8,M1_data9,M1_data10,M1_data11,M1_data12,M1_data13,M1_data14,M1_data15,M1_data16= [[] for x in range(classnum)]
    M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7,M1_train8, M1_train9, M1_train10, M1_train11, M1_train12, M1_train13, M1_train14,M1_train15, M1_train16 = [[] for x in range(classnum)]
    #M1_data:For each type of sample, the data structure: the structure, is used for the next sampling
    #M1_train1:For the training sample, the data structure: the structure, M1_data a part
    #M1_test:For the test sample, the data structure: the structure, M1_data part
    #Wholedata:For all samples, including the background, is used to generate the final classification chart
    M1_test=[]
    Wholedata = []
    # 首先定义一个类，要有__init__
    class SN:
        def __init__(self):
            self.Label = ''
            self.coordinate = []
            self.data = []
    ##获取除背景的所有样本分类别，为下一步抽取训练样本准备
    for row in range(y.shape[0]):
        for col in range(y.shape[1]):

            if y[row, col] == 0:
                struct1 = SN()
                struct1.Label = '0'
                struct1.coordinate.append((row, col))
                struct1.data.append(X[row, col, :])
                Wholedata.append(struct1)
            if y[row, col] == 1:
                struct1 = SN()
                struct1.Label='1'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col,:])
                M1_data1.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 2:
                struct1 = SN()
                struct1.Label='2'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col,:])
                M1_data2.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 3:
                struct1 = SN()
                struct1.Label='3'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col,:])
                M1_data3.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 4:
                struct1 = SN()
                struct1.Label='4'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col,:])
                M1_data4.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 5:
                struct1 = SN()
                struct1.Label='5'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data5.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 6:
                struct1 = SN()
                struct1.Label='6'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data6.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 7:
                struct1 = SN()
                struct1.Label='7'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data7.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 8:
                struct1 = SN()
                struct1.Label='8'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data8.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 9:
                struct1 = SN()
                struct1.Label='9'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data9.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 10:
                struct1 = SN()
                struct1.Label='10'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data10.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 11:
                struct1 = SN()
                struct1.Label='11'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data11.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 12:
                struct1 = SN()
                struct1.Label='12'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data12.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 13:
                struct1 = SN()
                struct1.Label='13'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data13.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 14:
                struct1 = SN()
                struct1.Label='14'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data14.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 15:
                struct1 = SN()
                struct1.Label='15'
                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data15.append(struct1)
                Wholedata.append(struct1)
            if y[row, col] == 16:
                struct1 = SN()
                struct1.Label='16'

                struct1.coordinate.append((row,col))
                struct1.data.append(X[row, col, :])
                M1_data16.append(struct1)
                print()
                Wholedata.append(struct1)

    trainsize = 10

    for i in range(classnum):
        initseedcoordinate = []#收集标注在图上
        for j in range(trainsize):
            length = len(locals()['M1_data{0}'.format(i + 1)])
            seedindex = random.randint(0, length-1)
            seed = locals()['M1_data{0}'.format(i+1)].pop(seedindex)
            locals()['M1_train{0}'.format(i+1)].append(seed)
            initseedcoordinate.append(seed.coordinate)

        Drawimage(image, initseedcoordinate, i+1)#
        M1_test.append(locals()['M1_data{0}'.format(i+1)])

    for iteration in range(15):
        #
        with open('../../Data/Salinas\\train{0}.txt'.format(iteration), 'w') as train:
            for i in range(classnum):
                for j in range(len(locals()['M1_train{0}'.format(i+1)])):
                    write_seed = locals()['M1_train{0}'.format(i+1)][j]

                    train.write('{0}'.format(i + 1) + "," + '{0}'.format(write_seed.coordinate[0]) + '\n')
        train.close()


        predictlabel,M1_testdata, M1_testcoordinate,probabilitylabel = Outputlabel_probaility(M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7,M1_train8, M1_train9, M1_train10, M1_train11, M1_train12, M1_train13, M1_train14,M1_train15, M1_train16, M1_test)

        combation_train = Combation(M1_train1, M1_train2, M1_train3, M1_train4, M1_train5, M1_train6, M1_train7,M1_train8, M1_train9, M1_train10, M1_train11, M1_train12, M1_train13, M1_train14,M1_train15, M1_train16, classnum)#合并训练样本用以空间约束

        #Obtain candidate sample data, candidate sample coordinates, candidate sample probability
        M1_candidatedata1, M1_candidatedata2, M1_candidatedata3, M1_candidatedata4, M1_candidatedata5, M1_candidatedata6, M1_candidatedata7,M1_candidatedata8, M1_candidatedata9, M1_candidatedata10, M1_candidatedata11, M1_candidatedata12, M1_candidatedata13, M1_candidatedata14,M1_candidatedata15, M1_candidatedata16,M1_candidatecoordinate1, M1_candidatecoordinate2, M1_candidatecoordinate3, M1_candidatecoordinate4, M1_candidatecoordinate5, M1_candidatecoordinate6, M1_candidatecoordinate7, M1_candidatecoordinate8, M1_candidatecoordinate9, M1_candidatecoordinate10, M1_candidatecoordinate11, M1_candidatecoordinate12, M1_candidatecoordinate13, M1_candidatecoordinate14, M1_candidatecoordinate15, M1_candidatecoordinate16,M1_candidatepro1, M1_candidatepro2, M1_candidatepro3, M1_candidatepro4, M1_candidatepro5, M1_candidatepro6, M1_candidatepro7, M1_candidatepro8, M1_candidatepro9, M1_candidatepro10, M1_candidatepro11, M1_candidatepro12, M1_candidatepro13, M1_candidatepro14, M1_candidatepro15, M1_candidatepro16=Labelclass(predictlabel, M1_testdata, M1_testcoordinate, probabilitylabel)
        #Obtain the first and second probability differences for each class of candidate samples
        M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16 = Calcuate_uncertain(M1_candidatepro1, M1_candidatepro2, M1_candidatepro3, M1_candidatepro4, M1_candidatepro5,M1_candidatepro6, M1_candidatepro7, M1_candidatepro8, M1_candidatepro9, M1_candidatepro10,M1_candidatepro11, M1_candidatepro12, M1_candidatepro13, M1_candidatepro14, M1_candidatepro15,M1_candidatepro16, classnum)
        #Normalization of probability difference
        M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16=Normalization(M1_prodiffer1,M1_prodiffer2,M1_prodiffer3,M1_prodiffer4,M1_prodiffer5,M1_prodiffer6,M1_prodiffer7,M1_prodiffer8,M1_prodiffer9,M1_prodiffer10,M1_prodiffer11,M1_prodiffer12,M1_prodiffer13,M1_prodiffer14,M1_prodiffer15,M1_prodiffer16, classnum)
        #Calculate data density
        M1_infodata1,M1_infodata2,M1_infodata3,M1_infodata4,M1_infodata5,M1_infodata6,M1_infodata7,M1_infodata8,M1_infodata9,M1_infodata10,M1_infodata11,M1_infodata12,M1_infodata13,M1_infodata14,M1_infodata15,M1_infodata16=Calcuate_information(M1_candidatedata1, M1_candidatedata2, M1_candidatedata3, M1_candidatedata4, M1_candidatedata5, M1_candidatedata6, M1_candidatedata7,M1_candidatedata8, M1_candidatedata9, M1_candidatedata10, M1_candidatedata11, M1_candidatedata12, M1_candidatedata13, M1_candidatedata14,M1_candidatedata15, M1_candidatedata16, classnum)
        #Data density normalization
        M1_infodata1,M1_infodata2,M1_infodata3,M1_infodata4,M1_infodata5,M1_infodata6,M1_infodata7,M1_infodata8,M1_infodata9,M1_infodata10,M1_infodata11,M1_infodata12,M1_infodata13,M1_infodata14,M1_infodata15,M1_infodata16=Normalization(M1_infodata1,M1_infodata2,M1_infodata3,M1_infodata4,M1_infodata5,M1_infodata6,M1_infodata7,M1_infodata8,M1_infodata9,M1_infodata10,M1_infodata11,M1_infodata12,M1_infodata13,M1_infodata14,M1_infodata15,M1_infodata16,classnum)
        #Calculate the score weight
        score1, score2, score3, score4, score5, score6, score7, score8, score9, score10, score11, score12, score13, score14, score15, score16=Calcuate_score(M1_prodiffer1, M1_prodiffer2, M1_prodiffer3, M1_prodiffer4, M1_prodiffer5, M1_prodiffer6,M1_prodiffer7, M1_prodiffer8, M1_prodiffer9, M1_prodiffer10, M1_prodiffer11, M1_prodiffer12,M1_prodiffer13, M1_prodiffer14, M1_prodiffer15, M1_prodiffer16, M1_infodata1, M1_infodata2,M1_infodata3, M1_infodata4, M1_infodata5, M1_infodata6, M1_infodata7, M1_infodata8, M1_infodata9,M1_infodata10, M1_infodata11, M1_infodata12, M1_infodata13, M1_infodata14, M1_infodata15,M1_infodata16, classnum)
        pickseed1,pickseed2,pickseed3,pickseed4,pickseed5,pickseed6,pickseed7,pickseed8,pickseed9,pickseed10,pickseed11,pickseed12,pickseed13,pickseed14,pickseed15,pickseed16 = [[] for x in range(classnum)]
        All_pickseedcoordinate = []#All selected samples were collected for culling in the test set
        for i in range(classnum):
            locals()['pickseed{0}'.format(i+1)] = Pickseed(locals()['score{0}'.format(i+1)], locals()['M1_candidatecoordinate{0}'.format(i+1)],  combation_train, i+1, classnum)
            # print(locals()['pickseed{0}'.format(i + 1)])

            Drawpickseedcoordinate = []

            for j in range(len(locals()['pickseed{0}'.format(i+1)])):
                pickseedlabel = str(i+1)
                seedindex = locals()['pickseed{0}'.format(i+1)][j]
                pickseedcoordinate = locals()['M1_candidatecoordinate{0}'.format(i+1)][seedindex]

                pickseeddata = [locals()['M1_candidatedata{0}'.format(i+1)][seedindex]]

                struct1 = SN()

                struct1.Label = pickseedlabel
                pickseedcoordinatex = pickseedcoordinate[0]
                pickseedcoordinatey = pickseedcoordinate[1]
                pick = [(pickseedcoordinatex, pickseedcoordinatey)]
                Drawpickseedcoordinate.append(pick)
                All_pickseedcoordinate.append(pick)
                # struct1.coordinate=[(pickseedcoordinatex,pickseedcoordinatey)]
                struct1.coordinate =pick
                struct1.data=pickseeddata

                locals()['M1_train{0}'.format(i+1)].append(struct1)
            Drawimage(image, Drawpickseedcoordinate, i + 1)
        #Select samples are excluded from the test sample
        M1_test = Remove(M1_test, All_pickseedcoordinate, classnum)



        print("1")

















