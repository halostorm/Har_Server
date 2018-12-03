import pickle
import os

import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import FeatureExtract as Ft


class Xgb_clf:
    xgb_model = xgb.XGBClassifier()
    model_path = r'xgb.pickle'

    def __init__(self, model_path=r'xgb.pickle'):
        self.model_path = model_path

    def train(self, x_train, y_train):
        # xgb分类器
        parameters = {
             #'nthread':[1],
            'objective': ['binary:logistic'],
            # 'learning_rate':[0.05],
            # 'learning_rate':[0.05,0.01,0.005,0.07,0.10],
            'learning_rate': [0.10,0.20],
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
        print(test_result_xgb)
        test_acc_xgb = accuracy_score(y_test, test_result_xgb)
        print('xgb test accuracy:', test_acc_xgb)

    def module_store(self, clf):
        with open(self.model_path, 'wb') as fw:
            pickle.dump(clf, fw)

    def module_load(self):
        with open(self.model_path, 'rb') as fr:
            self.xgb_model = pickle.load(fr)

    def inference(self,features):
        data = []
        for item in features:
            data.append((float)(item))
        input = list(reversed(data))
        # input = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
        input= np.reshape(input,[1,len(input)])
        print("ok2")
        res = self.xgb_model.predict(input)
        return res

def readData(FilePath):
    accNorm = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            tmp = np.sqrt((float)(line[2]) ** 2 + (float)(line[3]) ** 2 + (float)(line[4]) ** 2) - 9.806
            if i%2==0:
                accNorm.append(tmp)
            i+=1
    return accNorm


def extractFeatures(accNorm, state, featurePath=r'../traindata.txt'):
    N = (int)(len(accNorm) / 256)
    featureExtract = Ft.FeatureExtract(256, 50)
    with open(featurePath, 'a+')as fr:
        # Save Features
        for i in range(N):
            featureExtract.inputData(accNorm[256 * i:256 * (i + 1)])
            featureExtract.Features = []
            featureExtract.timeDomainFeature()
            featureExtract.frequencyDomainFeature()
            featureStream = featureExtract.Features
            fr.write(str(state) + ' ')
            for i in range(len(featureStream)):
                fr.write(str(i+1) + ':' + str(featureStream[i]) + ' ')
            fr.write('\n')


def generateTrainData(FilePath):
    for root, dirs, fileName in os.walk(FilePath):
        for i in fileName:
            state = 0
            dataDir = os.path.join(root, i)
            if 'static' in dataDir:
                state = 1
            elif 'walk' in dataDir:
                state = 2
            elif 'run' in dataDir:
                state = 3
            elif 'ride' in dataDir:
                state = 4
            elif 'car' in dataDir:
                state = 5

            accNorm = readData(dataDir)
            extractFeatures(accNorm, state)
            print(str(state) + '\t' + dataDir)


if __name__ == '__main__':
    FilePath = r'../traindata/data/'
    generateTrainData(FilePath)

    Xgb_clf = Xgb_clf()
    x_train, y_train = load_svmlight_file(r'../traindata.txt')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))
    print(x_test)

    Xgb_clf.train(x_train,y_train)
    Xgb_clf.module_load()
    Xgb_clf.test(x_test,y_test)

