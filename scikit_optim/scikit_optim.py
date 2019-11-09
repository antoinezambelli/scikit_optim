"""
#
# scikit_optim.py
#
# Copyright (c) 2018 Antoine Emil Zambelli. MIT License.
#
"""

import numpy as np
import os
import pandas as pd
import sys
import time
import warnings

import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'  # For ConvergenceWarning in CVs.

done_list = None
todo_list = None
t_1 = None
t_0 = None
curr_model = None

CPU_USE = max(os.cpu_count() // 3, 1)  # 12 -> 4; 8 -> 2 (good for laptops).

class ModelSelector():
    def __init__(self, ignore=(), check=(), acc_metric='accuracy_score'):
        self.ignore_list = list(ignore)  # list of strings, models to ignore.
        self.check_list = list(check)  # list of strings, models to look at only.
        self.acc_metric = acc_metric  # scoring param to optimize on in CV.
        self.summary_df = None  # performance and runtime for each model.
        self.summary_df_cv = None  # in-sample performance version - best CV-search score.
        self.models = None  # dict of model objects
        self.params = None  # dict of all best params for all evaluated models.
        self.best_model = None  # string, name of best model.
        self.best_params = None  # dict of best params for the best model.

    def fit(self, X_in, y_in, X_te_in=None, y_te_in=None):
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
        summary_dict_cv = {model: 0 for model in todo_list}
        model_dict = {model: 0 for model in todo_list}
        params = {model: 0 for model in todo_list}  # stores params for each model.

        # loop over todo_list and score.
        for model in todo_list:
            t_0 = time.time()
            curr_model = model
            if model == 'GMM':
                mod = GMM(acc_metric=self.acc_metric)
            elif model == 'LogRegress':
                mod = LogRegress(acc_metric=self.acc_metric)
            elif model == 'DecTree':
                mod = DecTree(acc_metric=self.acc_metric)
            elif model == 'RandForest':
                mod = RandForest(acc_metric=self.acc_metric)
            elif model == 'SupportVC':
                mod = SupportVC(acc_metric=self.acc_metric)
            elif model == 'kNN':
                mod = kNN(acc_metric=self.acc_metric)
            elif model == 'GaussNB':
                mod = GaussNB(acc_metric=self.acc_metric)
            elif model == 'MultiNB':
                mod = MultiNB(acc_metric=self.acc_metric)

            if X_te_in and y_te_in:
                mod_score = round(mod.score(X_in, y_in, X_te_in, y_te_in) * 100,  2)  # OoS score case.
            else:
                mod_score = round(mod.fit(X_in, y_in).best_score * 100,  2)  # In-sample score case.

            summary_dict[model] = {
                'time': time.strftime("%H:%M:%S",  time.gmtime(time.time()-t_0)),
                self.acc_metric: mod_score
            }  # Will still be defined for in-smaple case, but maybe not as meaningfully.
            summary_dict_cv[model] = {
                'time': time.strftime("%H:%M:%S",  time.gmtime(time.time()-t_0)),
                self.acc_metric: mod.best_score
            }

            params[model] = mod.best_params
            model_dict[model] = mod

            done_list.append(model)

        # get df and sort based on perf. store bests.
        summ_df = pd.DataFrame.from_dict(summary_dict, orient='index')
        summ_df = summ_df.sort_values(by=[self.acc_metric], ascending=False)

        summ_df_cv = pd.DataFrame.from_dict(summary_dict_cv, orient='index')
        summ_df_cv = summ_df_cv.sort_values(by=[self.acc_metric], ascending=False)

        self.best_model = summ_df.index[0]
        self.best_params = params[self.best_model]
        self.params = params
        self.summary_df = summ_df
        self.summary_df_cv = summ_df_cv
        self.models = model_dict

        return self


class GaussNB():
    def __init__(self, acc_metric='accuracy_score'):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_params = None
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'var_smoothing': np.geomspace(1e-9, 1e-3, 10)
            }
            gnb = GaussianNB()
            clf = GridSearchCV(
                gnb,
                n_jobs=CPU_USE,
                param_grid=parameters,
                scoring=self.acc_metric.split('_score')[0],
                cv=5
            )
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = GaussianNB(var_smoothing=best_params['var_smoothing']).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = GaussianNB(var_smoothing=best_params['var_smoothing']).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class MultiNB():
    def __init__(self, acc_metric='accuracy_score'):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_params = None
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'alpha': np.linspace(0,1,11)  # [0.1, 0.2, ..., 1.0].
            }
            mnb = MultinomialNB()
            clf = GridSearchCV(
                mnb,
                n_jobs=CPU_USE,
                param_grid=parameters,
                scoring=self.acc_metric.split('_score')[0],
                cv=5
            )
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = MultinomialNB(alpha=best_params['alpha']).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = MultinomialNB(alpha=best_params['alpha']).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class kNN():
    def __init__(self, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'n_neighbors': [
                    val
                    for val in [3, 5, 7, 9]
                ]
            }
            knn = KNeighborsClassifier()
            clf = GridSearchCV(
                knn,
                n_jobs=CPU_USE,
                param_grid=parameters,
                scoring=self.acc_metric.split('_score')[0],
                cv=5
            )
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

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

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class SupportVC():
    def __init__(self, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'kernel': ['rbf', 'sigmoid', 'linear'],
                'gamma': np.arange(0.2, 1.0, 0.2),
                'C': np.geomspace(0.01, 100, num=5)
            }

            svc = SVC()
            clf = GridSearchCV(svc, n_jobs=CPU_USE, param_grid=parameters, scoring=self.acc_metric.split('_score')[0], cv=5)
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

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

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class RandForest():
    def __init__(self, num_iter=200, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'min_samples_split': [2, 5, 8],
                'max_features': ['auto', None],
                'n_estimators': [10, 50]
            }
            rf = RandomForestClassifier()
            clf = RandomizedSearchCV(
                rf,
                n_jobs=CPU_USE,
                param_distributions=parameters,
                scoring=self.acc_metric.split('_score')[0],
                n_iter=self.num_iter,
                cv=5
            )

            try:
                clf.fit(X, Y)
            except ValueError:
                clf = GridSearchCV(
                    rf,
                    n_jobs=CPU_USE,
                    param_grid=parameters,
                    scoring=self.acc_metric.split('_score')[0],
                    cv=5
                )  # triggers if space is < num_iter.
                clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

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

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class DecTree():
    def __init__(self, num_iter=2500, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'min_samples_split': np.unique(np.round(np.geomspace(2, min(len(X)/100.0, 25), num=10))).astype(int),
                'max_features': np.unique(np.round(np.geomspace(np.sqrt(X_in.shape[1]), X_in.shape[1], num=5))).astype(int)
            }
            dc = DecisionTreeClassifier()
            clf = RandomizedSearchCV(
                dc,
                n_jobs=CPU_USE,
                param_distributions=parameters,
                scoring=self.acc_metric.split('_score')[0],
                n_iter=self.num_iter,
                cv=5
            )

            try:
                clf.fit(X, Y)
            except ValueError:
                clf = GridSearchCV(
                    dc,
                    n_jobs=CPU_USE,
                    param_grid=parameters,
                    scoring=self.acc_metric.split('_score')[0],
                    cv=5
                )
                clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

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

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class LogRegress():
    def __init__(self, num_iter=300, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            solver_list = ['lbfgs', 'newton-cg', 'saga']
            parameters = {'solver': solver_list, 'C': np.geomspace(0.01, 100, 5)}

            logreg = LogisticRegression()
            clf = GridSearchCV(
                logreg,
                n_jobs=CPU_USE,
                param_grid=parameters,
                scoring=self.acc_metric.split('_score')[0],
                cv=5
            )
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = LogisticRegression(C=best_params['C'], solver=best_params['solver']).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return y_pred

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = LogisticRegression(
            C=best_params['C'],
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

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score


class GMM():
    def __init__(self, best_params=None, acc_metric='accuracy_score'):
        self.best_params = best_params
        self.y_pred = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0

    def fit(self, X_in, Y_in):
        X = X_in.copy()
        Y = Y_in.copy()

        if self.best_params:
            pass
        else:
            parameters = {
                'n_components': [2],
                'covariance_type': ['full', 'diag', 'spherical', 'tied']
            }

            gmm = GaussianMixture()
            clf = GridSearchCV(
                gmm,
                n_jobs=CPU_USE,
                param_grid=parameters,
                scoring=self.acc_metric.split('_score')[0],
                cv=5
            )
            clf.fit(X, Y)

            self.best_params = clf.best_params_
            self.best_score = clf.best_score_

        return self

    def predict(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred = GaussianMixture(
            n_components=best_params['n_components'],
            covariance_type=best_params['covariance_type']
        ).fit(X_tr, Y_tr).predict(X_oos)

        self.y_pred = y_pred

        return res_df.values

    def predict_proba(self, X_in, y_in, X_te_in):
        X_tr = X_in.copy()
        Y_tr = y_in.copy()
        X_oos = X_te_in.copy()

        best_params = self.best_params

        y_pred_prob = GaussianMixture(
            n_components=best_params['n_components'],
            covariance_type=best_params['covariance_type']
        ).fit(X_tr, Y_tr).predict_proba(X_oos)

        self.y_pred_prob = y_pred_prob
        self.label_prob = np.max(y_pred_prob, axis=1)

        return y_pred_prob

    def score(self, X_in, y_in, X_te_in, y_te_in):
        X = X_in.copy()
        y = y_in.copy()

        X_te = X_te_in.copy()
        y_te = y_te_in.copy()

        if self.acc_metric == 'average_precision_score':
            y_pred = self.fit(X, y).predict_proba(X, y, X_te)[:, 1]
        else:
            y_pred = self.fit(X, y).predict(X, y, X_te)
        acc_score = getattr(sklearn.metrics, self.acc_metric)(y_te, y_pred)

        self.accuracy_score = acc_score
        return acc_score
