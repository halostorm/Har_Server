import pickle
import os
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


class Xgb_clf:
    xgb_model = xgb.XGBClassifier()
    model_path = r'xgb.pickle'

    def __init__(self, model_path=r'xgb.pickle'):
        self.model_path = model_path

    def train(self, x_train, y_train, model_path):
        # xgb分类器
        parameters = {
            # 'nthread':[4],
            'objective': ['binary:logistic'],
            # 'learning_rate':[0.05],
            # 'learning_rate':[0.05,0.01,0.005,0.07,0.10],
            'learning_rate': [0.10],
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

        self.module_store(clf_xgb, model_path)

        print('xgb accuracy :', score)

    def load_model(self, modulename):
        self.xgb_model = self.module_load(modulename)

    def test(self, x_test, y_test):
        test_result_xgb = self.xgb_model.predict(x_test)
        test_acc_xgb = accuracy_score(y_test, test_result_xgb)
        print('xgb test accuracy:', test_acc_xgb)

    def module_store(self, clf, moduleName):
        with open(moduleName, 'wb') as fw:
            pickle.dump(clf, fw)

    def module_load(self, moduleFile):
        with open(moduleFile, 'rb') as fr:
            clf = pickle.load(fr)
        return clf
