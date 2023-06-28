import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc, cohen_kappa_score
from sklearn.metrics import confusion_matrix

# readpath = 'D:\Secondwork_2022_11\Experiment1\paviau\\'#root path
# files = os.listdir(readpath)
# # for i in range(len(files)):
# #     print(files[i])
# # print(readpath+str(files[0])+str('\\groundtruthmap3.png'))#result path
# for k in range(0, len(files)):
#     path = readpath+str(files[k])
#     files1 = os.listdir(path)
#     for file in files1:
#         if file.endswith('.bmp'):
#             print(path + str('\\') + file)
#
#             predictimg = cv2.imread(path + str('\\') + file)
#
#             truthimg = cv2.imread("D:\\Firstwork_2022_9\\Traditional\\Data\\paviaU\\pavaiU_gt.bmp")#groundtruth path
#             predict = []
#             truth = []
#             for i in range(predictimg.shape[0]):
#                 for j in range(predictimg.shape[1]):
#                     if truthimg[i, j, 0] != 0:
#                         truth.append(truthimg[i, j, 0])
#                         predict.append(predictimg[i, j, 0])
#
#             truth = np.asarray(truth)
#             predict = np.asarray(predict)
#             print(truth)
#             print(predict)
#
#             def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#                 """
#                 - cm : 计算出的混淆矩阵的值
#                 - classes : 混淆矩阵中每一行每一列对应的列
#                 - normalize : True:显示百分比, False:显示个数
#                 """
#                 if normalize:
#                     cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
#                     print("显示百分比：")
#                     np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
#                     print(cm)
#                 else:
#                     print('显示具体数字：')
#                     print(cm)
#                 aa = 0
#                 for i in range(len(cm)):
#                     aa = aa+cm[i,i]
#                 averageaccuracy = aa/(len(classes))
#                 sumsdua = 0
#                 for i in range(len(cm)):
#                     sumsdua = sumsdua+(averageaccuracy-cm[i,i])**2
#                 sdua = np.sqrt(sumsdua/len(classes))
#                 plt.imshow(cm, interpolation='nearest', cmap=cmap)
#                 plt.title(title)
#                 plt.colorbar()
#                 tick_marks = np.arange(len(classes))
#                 plt.xticks(tick_marks, classes, rotation=45)
#                 plt.yticks(tick_marks, classes)
#                 # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
#                 plt.ylim(len(classes) - 0.5, -0.5)
#                 fmt = '.4f' if normalize else 'd'
#                 thresh = cm.max() / 2.
#                 for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#                     plt.text(j, i, format(cm[i, j], fmt),
#                              horizontalalignment="center",
#                              color="white" if cm[i, j] > thresh else "black")
#                 plt.tight_layout()
#                 plt.ylabel('True label')
#                 plt.xlabel('Predicted label')
#                 # plt.savefig(readpath + str('\\iteration9\\Oversample_Confusion_Matrix_train1.png'))
#                 plt.close()
#                 return averageaccuracy, sdua, cm
#
#
#             cnf_matrix = np.array(confusion_matrix(truth, predict, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
#
#             attack_types = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
#             averageaccuracy, sdua, cm= plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')
#
#             def record_output(oa_ae, kappa_ae, sdua, precision, recall, fscore,k, path):
#
#                 f = open(path, 'a')
#                 f.write('-------------------------------------------------------------------------------------------------------------'+'\n')
#                 f.write('OA:整体精度'+ '\n')
#                 f.write('Kappa:kappa系数' + '\n')
#                 f.write('AA:平均用户精度' + '\n')
#                 f.write('SDUA:平衡指标' + '\n')
#                 f.write('precision:平均精确率'+ '\n')
#                 f.write('recall:召回率'+ '\n')
#                 f.write('Fscore:F1分数'+ '\n')
#                 f.write(readpath+str(files[k])+str('\\{}'.format(file))+ '\n')
#                 f.write('\n')
#                 f.write("横向为truth，纵向为prediction："+'\n')
#                 for i in range(9):
#                     for j in range(9):
#                         f.write(str(cnf_matrix[i][j])+'\t')
#                     f.write('\n')
#                 for i in range(9):
#                     for j in range(9):
#                         f.write(str('%.4f'%cm[i][j]) + '\t')
#                     f.write('\n')
#                 f.write('\n')
#
#                 sentence0 = 'OA is:' + str('%.4f' % oa_ae) + '\n'
#                 f.write(sentence0)
#                 sentence1 = 'AA is:' + str('%.4f' % precision) + '\n'
#                 f.write(sentence1)
#                 sentence2 = 'KAPPA is:' + str('%.4f' % kappa_ae) + '\n'
#                 f.write(sentence2)
#                 sentence3 = 'SDUA is:' + str('%.4f' % sdua) + '\n'
#                 f.write(sentence3)
#                 sentence3 = 'precision is:' + str('%.4f' % precision) + '\n'
#                 f.write(sentence3)
#                 sentence3 = 'recall is:' + str('%.4f' % recall) + '\n'
#                 f.write(sentence3)
#                 sentence3 = 'Fscore is:' + str('%.4f' % fscore) + '\n'
#                 f.write(sentence3)
#                 f.write('-------------------------------------------------------------------------------------------------------------'+'\n')
#                 f.close()
#
#
#
#
#
#             acccuracy = float(accuracy_score(truth, predict))
#             #
#             Fscore = float(f1_score(truth, predict, average='macro'))
#             precision = float(precision_score(truth, predict, average='macro'))
#             recall = float(recall_score(truth, predict, average='macro'))
#             # print("over_accuracy:%f" % (acccuracy))
#             # print("average_accuracy:%f" % (averageaccuracy))
#             # overerror = float(1 - acccuracy)
#             kppa = cohen_kappa_score(truth, predict)
#             # print("overerror:%f" % (overerror))
#             # print("precision:%f" % (precision))
#             # print("recall:%f" % (recall))
#             # print("Fscore:%f" % (Fscore))
#             # print("kppa:%f"%(kppa))
#             # print("sdua:%f"%(sdua))
#
#             record_output(
#                 acccuracy, kppa, sdua,precision,recall,Fscore,k, 'D:\Secondwork_2022_11\Experiment1\Evaluationresult.txt')



#WHU
readpath = 'H:\CNN_Enhanced_GCN-master\\'#根路径
files = os.listdir(readpath)

for k in range(0, len(files)):

    if files[k].endswith('.bmp'):
        print(readpath + str('\\') + files[k])

        predictimg = cv2.imread(readpath + str('\\') + files[k])

        truthimg = cv2.imread("../../Data/WHU_Longkou/WHU_Longkou_gt.bmp")#对比真值路径
        predict = []
        truth = []
        for i in range(predictimg.shape[0]):
            for j in range(predictimg.shape[1]):
                if truthimg[i, j, 0] != 0:
                    truth.append(truthimg[i, j, 0])
                    predict.append(predictimg[i, j, 0])

        truth = np.asarray(truth)
        predict = np.asarray(predict)
        print(truth)
        print(predict)

        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            - cm : 计算出的混淆矩阵的值
            - classes : 混淆矩阵中每一行每一列对应的列
            - normalize : True:显示百分比, False:显示个数
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
                print("显示百分比：")
                np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
                print(cm)
            else:
                print('显示具体数字：')
                print(cm)
            aa = 0
            for i in range(len(cm)):
                aa = aa+cm[i,i]
            averageaccuracy = aa/(len(classes))
            sumsdua = 0
            for i in range(len(cm)):
                sumsdua = sumsdua+(averageaccuracy-cm[i,i])**2
            sdua = np.sqrt(sumsdua/len(classes))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
            plt.ylim(len(classes) - 0.5, -0.5)
            fmt = '.4f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.savefig(readpath + str('\\iteration9\\Oversample_Confusion_Matrix_train1.png'))
            plt.close()
            return averageaccuracy, sdua, cm


        cnf_matrix = np.array(confusion_matrix(truth, predict, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

        attack_types = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        averageaccuracy, sdua, cm= plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')

        def record_output(oa_ae, kappa_ae, sdua, precision, recall, fscore,k, path):

            f = open(path, 'a')
            f.write('-------------------------------------------------------------------------------------------------------------'+'\n')
            f.write('OA:整体精度'+ '\n')
            f.write('Kappa:kappa系数' + '\n')
            f.write('AA:平均用户精度' + '\n')
            f.write('SDUA:平衡指标' + '\n')
            f.write('precision:平均精确率'+ '\n')
            f.write('recall:召回率'+ '\n')
            f.write('Fscore:F1分数'+ '\n')
            f.write(readpath+str(files[k])+str('\\{}'.format(files[k]))+ '\n')
            f.write('\n')
            f.write("横向为truth，纵向为prediction："+'\n')
            for i in range(9):
                for j in range(9):
                    f.write(str(cnf_matrix[i][j])+'\t')
                f.write('\n')
            for i in range(9):
                for j in range(9):
                    f.write(str('%.4f'%cm[i][j]) + '\t')
                f.write('\n')
            f.write('\n')

            sentence0 = 'OA is:' + str('%.4f' % oa_ae) + '\n'
            f.write(sentence0)
            sentence1 = 'AA is:' + str('%.4f' % precision) + '\n'
            f.write(sentence1)
            sentence2 = 'KAPPA is:' + str('%.4f' % kappa_ae) + '\n'
            f.write(sentence2)
            sentence3 = 'SDUA is:' + str('%.4f' % sdua) + '\n'
            f.write(sentence3)
            sentence3 = 'precision is:' + str('%.4f' % precision) + '\n'
            f.write(sentence3)
            sentence3 = 'recall is:' + str('%.4f' % recall) + '\n'
            f.write(sentence3)
            sentence3 = 'Fscore is:' + str('%.4f' % fscore) + '\n'
            f.write(sentence3)
            f.write('-------------------------------------------------------------------------------------------------------------'+'\n')
            f.close()





        acccuracy = float(accuracy_score(truth, predict))
            #
        Fscore = float(f1_score(truth, predict, average='macro'))
        precision = float(precision_score(truth, predict, average='macro'))
        recall = float(recall_score(truth, predict, average='macro'))
        # print("over_accuracy:%f" % (acccuracy))
        # print("average_accuracy:%f" % (averageaccuracy))
        # overerror = float(1 - acccuracy)
        kppa = cohen_kappa_score(truth, predict)
        # print("overerror:%f" % (overerror))
        # print("precision:%f" % (precision))
        # print("recall:%f" % (recall))
        # print("Fscore:%f" % (Fscore))
        # print("kppa:%f"%(kppa))
        # print("sdua:%f"%(sdua))

        record_output(
                acccuracy, kppa, sdua,precision,recall,Fscore,k, 'H:\CNN_Enhanced_GCN-master\Evaluationresult.txt')