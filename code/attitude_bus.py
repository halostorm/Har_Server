import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import time
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


def readAtt(FilePath, times, Yaw):
    ekfX = []
    ekfY = []
    ekfZ = []

    Q = []

    with open(FilePath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if (len(line) < 15):
                continue
            tt = float(line[1])
            id = findGps(tt, times)
            yaw = Yaw[id]
            yaw_Q = euler2q(yaw)
            ekfQ = []
            ekfQ.append(float(line[11]))
            ekfQ.append(float(line[12]))
            ekfQ.append(float(line[13]))
            ekfQ.append(float(line[14]))

            q_new = q_x(yaw_Q,ekfQ)
            q_new = yaw_Q
            Q.append(q_new)

            ekfEuler = Q2Euler(ekfQ)
            ekfX.append(ekfEuler[0])
            ekfY.append(ekfEuler[1])
            ekfZ.append(ekfEuler[2])
    Q = np.matrix(Q)

    return ekfX, ekfY, ekfZ,Q


def WriteEuler(Filepath, X, Y, Z):
    with open(Filepath, 'a+') as f:
        for i in range(len(X)):
            f.write(str(X[i]) + '\t' + str(Y[i]) + '\t' + str(Z[i]) + '\n')
        f.write('\n')


def transform(x, y, z, T):
    for i in range(len(x)):
        x_ = T[0, 0] * x[i] + T[0, 1] * y[i] + T[0, 2] * z[i]
        y_ = T[1, 0] * x[i] + T[1, 1] * y[i] + T[1, 2] * z[i]
        z_ = T[2, 0] * x[i] + T[2, 1] * y[i] + T[2, 2] * z[i]
        x[i] = x_
        y[i] = y_
        z[i] = z_
    return x, y, z


def findGps(t, tts):
    min = 10000000000000000
    id = 0
    for i in range(len(tts)):
        if min > (np.abs(t - tts[i])):
            min = np.abs(t - tts[i])
            id = i
    return id


def readGps(FilePath):
    time = []
    bear = []
    Ts = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            time.append(float(line[0]))
            euler = (float(line[4])) / 360.0 * 2 * 3.14159265
            bear.append(euler)
            Ts.append(euler2matrix(euler))
            i += 1
    return time, bear


def Q2Euler(q):
    eulerAngles = []
    eulerAngles.append(
        (float)(np.math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                              1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))))  # / np.pi * 180))
    eulerAngles.append((float)(
        math.asin(2.0 * (q[0] * q[2] - q[3] * q[1]))))
    eulerAngles.append((float)(
        math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))))
    return eulerAngles


def euler2matrix(e):
    T = np.mat(np.zeros((3, 3)))
    T[0, 0] = np.cos(e)
    T[0, 1] = -np.sin(e)

    T[1, 0] = np.sin(e)
    T[1, 1] = np.cos(e)

    T[2, 2] = 1
    return T.transpose()


def euler2q(e):
    q = []
    q.append(np.cos(e / 2))
    q.append(0)
    q.append(0)
    q.append(-np.sin(e / 2))
    return q


def q_x(q1, q2):
    q = [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]]
    return q


def graph(d1,d2,d3,d4):
    plt.figure()
    plt.plot(d1, c='k', lw=0.5)
    plt.plot(d2, c='r', lw=0.5)
    plt.plot(d3, c='g', lw=0.5)
    plt.plot(d4, c='b', lw=0.5)
    plt.show()


import random

import keras
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
import FeatureExtract as Ft
import os


class NN_clf:
    model = None
    n_classes = 3

    def __init__(self):
        model = Sequential()
        model.add(Dense(20, activation='relu', input_dim=17*4))
        model.add(BatchNormalization(axis=1, epsilon=1.1e-5))
        model.add(Dense(3, activation='softmax'))
        optimizer = Adam(lr=1e-2)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
        self.model = model

    def train(self, features, labels):
        print('train')
        nb_epoch = 2
        batch_size = 10

        random.seed(21)
        random.shuffle(features)
        random.seed(21)
        random.shuffle(labels)

        testX = []
        testY = []
        trainX = []
        trainY = []

        for i in range(len(features)):
            if i < 0.7 * len(features):
                trainX.append(features[i])
                trainY.append(labels[i])
            else:
                testX.append(features[i])
                testY.append(labels[i])

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        trainY = keras.utils.to_categorical(trainY, num_classes=self.n_classes)

        testX = np.array(testX)
        testY = np.array(testY)
        testY = keras.utils.to_categorical(testY, num_classes=self.n_classes)

        print(np.shape(trainX))
        print(np.shape(trainY))
        print(np.shape(testX))
        print(np.shape(testY))

        # 若训练过，加载预训练参数
        weights_file = r'attitude.h5'
        if os.path.exists(weights_file):
            self.model.load_weights(weights_file, by_name=True)
            print('Model loaded.')
        self.model.summary()

        # 若未训练过，则训练
        if not os.path.exists(weights_file):
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                           cooldown=0, patience=10, min_lr=0.5e-6)
            early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
            model_checkpoint = ModelCheckpoint(r'attitude.h5', monitor='val_acc', save_best_only=True,
                                               save_weights_only=True)

            callbacks = [lr_reducer, early_stopper, model_checkpoint]

            history = self.model.fit(trainX, trainY, nb_epoch=nb_epoch,
                           callbacks=callbacks, batch_size=batch_size,
                           validation_data=(testX, testY), verbose=1)

        yPreds = self.model.predict(testX)
        yPred = np.argmax(yPreds, axis=-1)
        print(yPred)
        yTrue = np.argmax(testY, axis=-1)
        print(yTrue)

        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        error = 100 - accuracy
        print("Accuracy : ", accuracy)
        print("Error : ", error)

        test_acc_xgb = accuracy_score(yTrue, yPred)

        print('static')
        print(metrics.precision_score(yTrue, yPred, labels=[0], average='micro'))  # 微平均，精确率
        print(metrics.recall_score(yTrue, yPred, labels=[0], average='micro'))  # 微平均，召回率
        print('calling')
        print(metrics.precision_score(yTrue, yPred, labels=[1], average='micro'))  # 微平均，精确率
        print(metrics.recall_score(yTrue, yPred, labels=[1], average='micro'))  # 微平均，召回率
        print('texting')
        print(metrics.precision_score(yTrue, yPred, labels=[2], average='micro'))  # 微平均，精确率
        print(metrics.recall_score(yTrue, yPred, labels=[2], average='micro'))  # 微平均，召回率

        print('xgb test accuracy:', test_acc_xgb)

        self.ROC(testX,testY)


    def ROC(self, x_test, y_test):
        # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
        y_score = self.model.predict_proba(x_test)
        # 1、调用函数计算micro类型的AUC
        y_one_hot = label_binarize(y_test, np.arange(3))
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
        plt.plot(fpr, tpr, c='b', lw=1.5, alpha=1, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title("Detecting Result of user texing and phone on user's ear an", fontsize=14)
        plt.show()


    def loadModel(self):
        weights_file = r'attitude.h5'
        if os.path.exists(weights_file):
            self.model.load_weights(weights_file, by_name=True)
            print('Model loaded.')
        self.model.summary()


    def inference(self,features):
        data = []
        for item in features:
            data.append((float)(item))
        data = list(reversed(data))
        input = np.array(data)
        input = np.reshape(input, [1, len(input)])
        res = self.model.predict(input)
        print(res)
        res = np.argmax(res, axis=1)
        return res


def extractFeatures(Q1,Q2,Q3,Q4, state):
    N = (int)(len(Q1) / 256)
    featureExtract = Ft.FeatureExtract(256, 50)
    featureX = []
    labelY = []

    for i in range(N):
        featureExtract.inputData(Q1[256 * i:256 * (i + 1)])
        featureExtract.Features = []
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream1 = featureExtract.Features

        featureExtract.inputData(Q2[256 * i:256 * (i + 1)])
        featureExtract.Features = []
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream2 = featureExtract.Features

        featureExtract.inputData(Q3[256 * i:256 * (i + 1)])
        featureExtract.Features = []
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream3 = featureExtract.Features

        featureExtract.inputData(Q4[256 * i:256 * (i + 1)])
        featureExtract.Features = []
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream4 = featureExtract.Features

        featureStream1.extend(featureStream2)
        featureStream1.extend(featureStream3)
        featureStream1.extend(featureStream4)
        featureX.append(featureStream1)
        labelY.append(state)

    return featureX, labelY


def readData(FilePath):
    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            if len(line)>2:
                # print(line[0])
                Q1.append(float(line[0]))
                Q2.append(float(line[1]))
                Q3.append(float(line[2]))
                Q4.append(float(line[3]))
            i += 1
    return Q1,Q2,Q3,Q4


def generateTrainData(FilePath):
    trainX = []
    trainY = []
    for root, dirs, fileName in os.walk(FilePath):
        for i in fileName:
            state = 0
            dataDir = os.path.join(root, i)
            if 'static' in dataDir:
                state = 0
            elif 'call' in dataDir:
                state = 1
            elif 'text' in dataDir:
                state = 2
            print(dataDir)

            Q1,Q2,Q3,Q4 = readData(dataDir)

            Q1 = dataNoise(Q1)
            Q2 = dataNoise(Q2)
            Q3 = dataNoise(Q3)
            Q4 = dataNoise(Q4)


            featureX1, labelY1 = extractFeatures(Q1,Q2,Q3,Q4, state)

            trainX.extend(featureX1)
            trainY.extend(labelY1)

            print(str(state) + '\t' + dataDir)
    print(np.shape(trainY))
    print(np.shape(trainX))

    return trainX, trainY

def dataNoise(accNorm):
    acc1 = []
    mu = 0
    sigma = 0.1
    scale = 5
    for i in range(scale):
        for item in accNorm:
            acc1.append(item + random.gauss(mu, sigma * i))
    return acc1


if __name__ == '__main__':
    # times, Yaw = readGps(r'/home/halo/Workspace/Har_Server/busData/call/Location_2018_12_14_11_27_51.txt')
    #
    # X, Y, Z,Q = readAtt(r'/home/halo/Workspace/Har_Server/busData/call/raw_2018_12_14_11_43_26.txt', times, Yaw)
    #
    # graph(Q.A.T[0],Q.A.T[1],Q.A.T[2],Q.A.T[3])

    # trainX, trainY =  generateTrainData(r'/home/halo/attitude/')
    #
    # clf = NN_clf()
    # clf.train(trainX, trainY)

    model = Sequential()
    model.add(Dense(6, activation='sigmoid', input_dim=12))
    # model.add(BatchNormalization(axis=1, epsilon=1.1e-5))
    model.add(Dense(3, activation='softmax'))
    optimizer = Adam(lr=1e-2)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
