import pickle
import os
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

import FeatureExtract as Ft

class Xgb_clf:
    xgb_model = xgb.XGBClassifier()
    model_path = r'xgb.pickle'

    def __init__(self, model_path=r'xgb.pickle'):
        self.model_path = model_path

    def train(self, x_train, y_train):
        # xgb分类器
        parameters = {
            # 'nthread':[1],
            'objective': ['binary:logistic'],
            # 'learning_rate':[0.05],
            # 'learning_rate':[0.05,0.01,0.005,0.07,0.10],
            'learning_rate': [0.10, 0.20],
            'max_depth': [6],
            # 'max_depth':[4,5,6,7,8,9],
            'min_child_weight': [11],
            'silent': [1],
            'subsample': [0.8],
            'colsample_bytree': [0.7],
            'n_estimators': [5],
            'missing': [0.0],
            # 'seed':[40,50,60,1337],
            'seed': [1337],
            # 'lambda':[0.1]
            'gamma': [0],
            'max_delta_step': [0]
        }
        clf_xgb = GridSearchCV(self.xgb_model, parameters,
                               scoring='accuracy', verbose=2, refit=True)
        clf_xgb.fit(x_train, y_train)

        best_parameters, score, _ = max(clf_xgb.grid_scores_, key=lambda x: x[1])

        self.module_store(clf_xgb)

        print('xgb accuracy :', score)

    def test(self, x_test, y_test):
        test_result_xgb = self.xgb_model.predict(x_test)
        print(np.shape(test_result_xgb))
        test_acc_xgb = accuracy_score(y_test, test_result_xgb)

        print(metrics.precision_score(y_test, test_result_xgb,labels=[4], average='micro'))  # 微平均，精确率

        print(metrics.recall_score(y_test, test_result_xgb,labels=[4], average='micro'))  # 微平均，精确率

        print('xgb test accuracy:', test_acc_xgb)

    def ROC(self,x_test, y_test):
        # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
        y_score = self.xgb_model.predict_proba(x_test)
        # 1、调用函数计算micro类型的AUC
        y_one_hot = label_binarize(y_test, np.arange(5))
        print('调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro'))
        # 2、手动计算micro类型的AUC
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        auc = metrics.auc(fpr, tpr)
        print('手动计算auc：', auc)
        # 绘图
        mpl.rcParams['font.sans-serif'] = u'SimHei'
        mpl.rcParams['axes.unicode_minus'] = False
        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(fpr, tpr, c='g', lw=1.5, alpha=1, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title("Classification Result of human activities", fontsize=14)
        plt.show()


    def module_store(self, clf):
        with open(self.model_path, 'wb') as fw:
            pickle.dump(clf, fw)

    def module_load(self):
        with open(self.model_path, 'rb') as fr:
            self.xgb_model = pickle.load(fr)

    def inference(self, features):
        data = []
        for item in features:
            data.append((float)(item))
        input = list(reversed(data))
        # input = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
        input = np.reshape(input, [1, len(input)])
        print("ok2")
        res = self.xgb_model.predict(input)
        return res


def readData(FilePath):
    accNorm = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            tmp = np.sqrt((float)(line[2]) ** 2 + (float)(line[3]) ** 2 + (float)(line[4]) ** 2) - 9.7938
            if i % 2 == 0:
                accNorm.append(tmp)
                wr(r'../Seg.txt',line)
            i += 1
    return accNorm

def wr(F,data):
    with open(F,'a+') as f:
        f.write(data[2]+'\t'+data[3]+'\t'+data[4]+'\n')


def dataNoise(accNorm):
    acc1 = []
    mu = 0
    sigma = 0.1
    scale = 2
    for i in range(scale):
        for item in accNorm:
            acc1.append(item + random.gauss(mu, sigma * i))
    return acc1


def extractFeatures(accNorm, state, featurePath=r'../traindata_xgb.txt'):
    N = (int)(len(accNorm) / 256)
    featureExtract = Ft.FeatureExtract(256, 50)
    with open(featurePath, 'a+')as fr:
        # Save Features
        for i in range(N):
            featureExtract.inputData(accNorm[256 * i:256 * (i + 1)], State=state)
            featureExtract.Features = []
            featureExtract.id = i
            featureExtract.timeDomainFeature()
            featureExtract.frequencyDomainFeature()
            featureStream = featureExtract.Features
            fr.write(str(state) + ' ')
            for i in range(len(featureStream)):
                fr.write(str(i + 1) + ':' + str(featureStream[i]) + ' ')
            fr.write('\n')


def generateTrainData(FilePath):
    for root, dirs, fileName in os.walk(FilePath):
        for i in fileName:
            state = 0
            dataDir = os.path.join(root, i)
            if 'static' in dataDir:
                state = 0
            elif 'walk' in dataDir:
                state = 1
            elif 'run' in dataDir:
                state = 2
            elif 'ride' in dataDir:
                state = 3
            elif 'car' in dataDir:
                state = 4

            accNorm = readData(dataDir)
            accNorm = dataNoise(accNorm)
            extractFeatures(accNorm, state)
            print(str(state) + '\t' + dataDir)


def showRawData(FilePath):
    for root, dirs, fileName in os.walk(FilePath):
        for i in fileName:
            state = 0
            dataDir = os.path.join(root, i)
            if 'static1' in dataDir:
                state = 0
                X, Y, Z = readRawData(dataDir)
                draw(X, Y, Z,str(state))
            elif 'walk1' in dataDir:
                state = 1
                X, Y, Z = readRawData(dataDir)
                draw(X, Y, Z, str(state))
            elif 'run1' in dataDir:
                state = 2
                X, Y, Z = readRawData(dataDir)
                draw(X, Y, Z, str(state))
            elif 'ride1' in dataDir:
                state = 3
                X, Y, Z = readRawData(dataDir)
                draw(X, Y, Z, str(state))
            elif 'car1' in dataDir:
                state = 4
                X, Y, Z = readRawData(dataDir)
                draw(X, Y, Z, str(state))



def readRawData(FilePath):
    X = []
    Y = []
    Z = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            if i % 2 == 0 and (i>5000 and i<8000):
                line = line.strip().split('\t')
                X.append((float)(line[2]))
                Y.append((float)(line[3]))
                Z.append((float)(line[4]))
            i += 1
    return X,Y,Z

def draw(X,Y,Z,name):
    plt.figure(name)
    x = np.linspace(0, 60, len(X))
    plt.plot(x,X, lw=1, c='r',label = "X-axis")
    plt.plot(x,Y, lw=1, c='b',label = "Y-axis")
    plt.plot(x,Z, lw=1, c='g',label = "Z-axis")
    plt.legend()  # 展示图例
    plt.xlabel('Times /(seconds)')  # 给 x 轴添加标签
    plt.ylabel('Acceleration /(m/s2)')  # 给 y 轴添加标签

    my_y_ticks = np.arange(-30, 30, 5)
    plt.yticks(my_y_ticks)
    plt.show()

if __name__ == '__main__':
    FilePath = r'../traindata/data/'
    #generateTrainData(FilePath)
    # showRawData(FilePath)

    # Xgb_clf = Xgb_clf()
    # x_train, y_train = load_svmlight_file(r'../traindata_xgb.txt')
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=71)
    #
    # print(np.shape(x_train))
    # print(np.shape(y_train))
    # print(np.shape(x_test))
    # print(np.shape(y_test))
    #
    # Xgb_clf.train(x_train, y_train)
    # Xgb_clf.module_load()
    # Xgb_clf.test(x_test, y_test)
    accNorm = readData(r'/home/halo/Workspace/Har_Server/traindata/走路 手上2/raw_2018_5_19_14_2_33.txt')
    extractFeatures(accNorm, 0)
