import random

import keras
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import FeatureExtract as Ft
import os


class NN_clf:
    model = None
    n_classes = 5

    def __init__(self):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_dim=17))
        model.add(BatchNormalization(axis=1, epsilon=1.1e-5))
        model.add(Dense(5, activation='softmax'))
        optimizer = Adam(lr=1e-2)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
        self.model = model

    def train(self, features, labels):
        print('train')
        nb_epoch = 30
        batch_size = 5

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
        weights_file = r'har.h5'
        if os.path.exists(weights_file):
            self.model.load_weights(weights_file, by_name=True)
            print('Model loaded.')
        self.model.summary()

        # 若未训练过，则训练
        if not os.path.exists(weights_file):
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                           cooldown=0, patience=10, min_lr=0.5e-6)
            early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
            model_checkpoint = ModelCheckpoint(r'har.h5', monitor='val_acc', save_best_only=True,
                                               save_weights_only=True)

            callbacks = [lr_reducer, early_stopper, model_checkpoint]

            self.model.fit(trainX, trainY, nb_epoch=nb_epoch,
                           callbacks=callbacks, batch_size=batch_size,
                           validation_data=(testX, testY), verbose=1)

            # self.model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX),
            #                     nb_epoch=nb_epoch,
            #                     callbacks=callbacks,
            #                     validation_data=(testX, testY),
            #                     nb_val_samples=testX.shape[0], verbose=1)

    def loadModel(self):
        weights_file = r'har.h5'
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


def extractFeatures(accNorm, state):
    N = (int)(len(accNorm) / 256)
    featureExtract = Ft.FeatureExtract(256, 50)
    featureX = []
    labelY = []

    for i in range(N):
        featureExtract.inputData(accNorm[256 * i:256 * (i + 1)])
        featureExtract.Features = []
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream = featureExtract.Features

        featureX.append(featureStream)
        labelY.append(state)

    return featureX, labelY


def readData(FilePath):
    accNorm = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            tmp = np.sqrt((float)(line[2]) ** 2 + (float)(line[3]) ** 2 + (float)(line[4]) ** 2) - 9.7938
            if i % 2 == 0:
                accNorm.append(tmp)
            i += 1
    return accNorm


def generateTrainData(FilePath):
    trainX = []
    trainY = []
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
            featureX, labelY = extractFeatures(accNorm, state)

            trainX.extend(featureX)
            trainY.extend(labelY)

            print(str(state) + '\t' + dataDir)
    print(np.shape(trainY))
    print(np.shape(trainX))

    return trainX, trainY


if __name__ == '__main__':
    FilePath = r'../traindata/data/'
    trainX, trainY = generateTrainData(FilePath)

    clf = NN_clf()
    clf.train(trainX, trainY)

    # data = np.random.random((10, 10))
    # labels = np.random.randint(10, size=(10, 1))
    #
    # print(data)
    # print(labels)
    # print(np.shape(data))
    # print(np.shape(labels))
