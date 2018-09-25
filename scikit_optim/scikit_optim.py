"""
#
# scikit_optim.py
#
# Copyright (c) 2018 Antoine Emil Zambelli. MIT License.
#
"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
from scipy.stats import randint
import time


done_list = None
todo_list = None
t_1 = None
t_0 = None
curr_model = None


class ModelSelector():
    def __init__(self, ignore=[], check=[]):
        self.ignore_list = ignore  # list of strings,  models to ignore.
        self.check_list = check  # list of strings,  models to look at only.
        self.summary_df = None  # performance and runtime for each model.
        self.models = None  # dict of model objects
        self.params = None  # dict of all best params for all evaluated models.
        self.best_model = None  # string,  name of best model.
        self.best_params = None  # dict of best params for the best model.

    def fit(self, X_in, y_in, X_te_in, y_te_in):
        global done_list
        global todo_list
        global t_1
        global t_0
        global curr_model

        # get lists organized and initialize dicts.
        done_list = []
        if not self.check_list:
            self.check_list = [
                'GMM', 'LogRegress', 'DecTree', 'RandForest',
                'SupportVC', 'kNN', 'BGMM', 'GaussNB', 'MultiNB'
            ]  # default is all models.

        todo_list = [x for x in self.check_list if x not in self.ignore_list]  # list of models to evaluate.
        summary_dict = {model: 0 for model in todo_list}  # eventually turn into df.
        model_dict = {model: 0 for model in todo_list}
        params = {model: 0 for model in todo_list}  # stores params for each model.

        # loop over todo_list and score.
        for model in todo_list:
            t_0 = time.time()
            curr_model = model
            if model == 'GMM':
                mod = GMM()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'LogRegress':
                mod = LogRegress()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'DecTree':
                mod = DecTree()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'RandForest':
                mod = RandForest()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'SupportVC':
                mod = SupportVC()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'kNN':
                mod = kNN()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'BGMM':
                mod = BGMM()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'GaussNB':
                mod = GaussNB()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)
            elif model == 'MultiNB':
                mod = MultiNB()
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)

            summary_dict[model] = {
                'time': time.strftime("%H:%M:%S",  time.gmtime(time.time()-t_0)),
                'accuracy': mod_score
            }

            params[model] = mod.best_params
            model_dict[model] = mod

            done_list.append(model)

        # get df and sort based on perf. store bests.
        summ_df = pd.DataFrame.from_dict(summary_dict, orient='index')
        summ_df = summ_df.sort_values(by=['accuracy'], ascending=False)

        self.best_model = summ_df.index[0]
        self.best_params = params[self.best_model]
        self.params = params
        self.summary_df = summ_df
        self.models = model_dict

        return self


class GaussNB():
    def __init__(self):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None
        self.best_params = None

    def fit(self, X_in, Y_in):
        pass

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        y_pred = GaussianNB().fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        y_pred_prob = GaussianNB().fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class MultiNB():
    def __init__(self):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None
        self.best_params = None

    def fit(self, X_in, Y_in):
        pass

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        y_pred = MultinomialNB().fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        y_pred_prob = MultinomialNB().fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class kNN():
    def __init__(self, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {'n_neighbors': np.arange(1, 27, 2)}  # TODO: fix this to use smarter range. geomspace() from 1 to len(data)/100? Might need to unique(int()) the geomspace output.
            knn = KNeighborsClassifier()
            clf = GridSearchCV(knn, parameters)
            clf.fit(X, Y)

            self.best_params = clf.best_params_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = KNeighborsClassifier(n_neighbors=best_params['n_neighbors']).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = KNeighborsClassifier(n_neighbors=best_params['n_neighbors']).fit(X_tr, Y_tr).predict_proba(X_oos)
        self.label_prob = np.max(y_pred_prob, axis=1)

        self.y_pred_prob = y_pred_prob

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class SupportVC():
    def __init__(self, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'kernel': ['rbf', 'sigmoid', 'linear'],
                'gamma': np.arange(0.1, 1.0, 0.1),
                'C': np.geomspace(0.01, 100, num=20)
            }

            svc = SVC()
            clf = GridSearchCV(svc, parameters)
            clf.fit(X, Y)

            self.best_params = clf.best_params_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = SVC(
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            C=best_params['C']
        ).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = SVC(
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            C=best_params['C']
        ).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class RandForest():
    def __init__(self, num_iter=200, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None
        self.num_iter = num_iter

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'min_samples_split': np.arange(2, 22, 2),  # TODO: fix this to use smarter range.
                'max_features': randint(1, len(X.columns.values)),
                'n_estimators': np.arange(10, 110, 10)  # TODO: fix this to use smarter range.
            }
            rf = RandomForestClassifier()
            clf = RandomizedSearchCV(rf, parameters, n_iter=self.num_iter)
            clf.fit(X, Y)

            self.best_params = clf.best_params_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = RandomForestClassifier(
            min_samples_split=best_params['min_samples_split'],
            max_features=best_params['max_features'],
            n_estimators=best_params['n_estimators']
        ).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = RandomForestClassifier(
            min_samples_split=best_params['min_samples_split'],
            max_features=best_params['max_features'],
            n_estimators=best_params['n_estimators']
        ).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class DecTree():
    def __init__(self, num_iter=2500, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None
        self.num_iter = num_iter

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            if len(range(2, 21))*len(range(1, len(X_in.columns.values))) <= 500:
                parameters = {'min_samples_split': range(2, 21), 'max_features': range(1, len(X_in.columns.values))}  # TODO: fix this to use smarter range. change if condition as needed.
                dc = DecisionTreeClassifier()
                clf = GridSearchCV(dc, parameters)
                clf.fit(X, Y)
            else:
                parameters = {'min_samples_split': range(2, 21), 'max_features': range(1, len(X_in.columns.values))}
                dc = DecisionTreeClassifier()
                clf = RandomizedSearchCV(dc, parameters, n_iter=self.num_iter)
                clf.fit(X, Y)

            self.best_params = clf.best_params_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = DecisionTreeClassifier(
            min_samples_split=best_params['min_samples_split'],
            max_features=best_params['max_features']
        ).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = DecisionTreeClassifier(
            min_samples_split=best_params['min_samples_split'],
            max_features=best_params['max_features']
        ).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class LogRegress():
    def __init__(self, num_iter=300, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.accuracy_score = None
        self.num_iter = num_iter

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            solver_list = ['lbfgs', 'newton-cg', 'saga']
            parameters = {'solver': solver_list, 'Cs': [int(self.num_iter/3.0)]}

            logreg = LogisticRegressionCV()
            clf = GridSearchCV(logreg,  parameters)
            clf.fit(X, Y)

            self.best_params = clf.best_params_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = LogisticRegressionCV(Cs=best_params['Cs'], solver=best_params['solver']).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = LogisticRegressionCV(
            Cs=best_params['Cs'],
            solver=best_params['solver']
        ).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class BGMM():
    def __init__(self, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.accuracy_score = None

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            unique_class_vals = Y.unique()

            parameters = {
                'n_components': np.arange(1, 27, 2),
                'covariance_type': ['full', 'diag', 'spherical', 'tied'],
                'weight_concentration_prior': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
            gmm_param_dict = {class_val: 0 for class_val in unique_class_vals}

            gmm = BayesianGaussianMixture(reg_covar=1)
            clf = GridSearchCV(gmm,  parameters)
            for class_val in unique_class_vals:
                clf.fit(X[Y == class_val], Y[Y == class_val])
                gmm_param_dict[class_val] = clf.best_params_

            self.best_params = gmm_param_dict

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        unique_class_vals = Y_tr.unique()

        best_params = self.best_params

        gmm_dict = {class_val: 0 for class_val in unique_class_vals}
        for class_val in unique_class_vals:
            gmm = BayesianGaussianMixture(
                n_components=best_params[class_val]['n_components'],
                covariance_type=best_params[class_val]['covariance_type'],
                weight_concentration_prior=best_params[class_val]['weight_concentration_prior'],
                reg_covar=1
            ).fit(X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val])

            gmm_dict[class_val] = np.exp(gmm.score_samples(X_oos))

        gmm_df = pd.DataFrame.from_dict(gmm_dict)
        res_df = gmm_df.idxmax(axis=1)

        self.y_pred = res_df.values

        return res_df.values

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score


class GMM():
    def __init__(self, best_params=None):
        self.best_params = best_params
        self.y_pred = None
        self.accuracy_score = None

    def fit(self, X_in, Y_in):
        X_tr = X_in.copy()
        Y_tr = Y_in.copy()

        if self.best_params:
            pass
        else:
            unique_class_vals = Y_tr.unique()

            n_comp_list = range(1, 26)  # this may require try catch below.
            covar_list = ['full', 'diag', 'spherical', 'tied']

            gmm_param_dict = {class_val: {
                (n_comp, cov): GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cov
                ).fit(
                    X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val]
                ).bic(
                    X_tr[Y_tr == class_val]
                ) for n_comp in n_comp_list for cov in covar_list
            } for class_val in unique_class_vals}

            gmm_dict = {class_val: 0 for class_val in unique_class_vals}
            for class_val in unique_class_vals:
                gmm_dict[class_val] = min(gmm_param_dict[class_val],  key=gmm_param_dict[class_val].get)

            self.best_params = gmm_dict

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params
        unique_class_vals = best_params.keys()

        gmm_dict = {class_val: 0 for class_val in unique_class_vals}
        for class_val in unique_class_vals:
            try:
                gmm = GaussianMixture(
                    n_components=best_params[class_val][0],
                    covariance_type=best_params[class_val][1]
                ).fit(X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val])
            except:
                gmm = GaussianMixture(
                    n_components=best_params[class_val][0],
                    covariance_type=best_params[class_val][1],
                    reg_covar=1
                ).fit(X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val])
            gmm_dict[class_val] = np.exp(gmm.score_samples(X_oos))

        gmm_df = pd.DataFrame.from_dict(gmm_dict)
        res_df = gmm_df.idxmax(axis=1)
        self.y_pred = res_df.values

        return res_df.values

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = len(np.where(y_pred == y_te)[0])/float(len(y_te))

        self.accuracy_score = acc_score
        return acc_score

    def score_samples(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params
        unique_class_vals = best_params.keys()

        gmm_dict = {class_val: 0 for class_val in unique_class_vals}
        for class_val in unique_class_vals:
            try:
                gmm = GaussianMixture(
                    n_components=best_params[class_val][0],
                    covariance_type=best_params[class_val][1]
                ).fit(X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val])
            except:
                gmm = GaussianMixture(
                    n_components=best_params[class_val][0],
                    covariance_type=best_params[class_val][1],
                    reg_covar=1
                ).fit(X_tr[Y_tr == class_val], Y_tr[Y_tr == class_val])
            gmm_dict[class_val] = np.exp(gmm.score_samples(X_oos))

        return gmm_dict
