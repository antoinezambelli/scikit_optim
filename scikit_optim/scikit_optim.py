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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


if not sys.warnoptions:
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'  # For ConvergenceWarning in CVs.

CPU_ALLOC = {24: 8, 12: 4, 8: 2, 4: 3}  # 8->2 good for laptops, 4->3 for RPis.
CPU_USE = CPU_ALLOC.get(os.cpu_count(), 1)

def bucket_data(df, bucket_list=None):
    '''
    Convenience method: this bins features.
    '''
    if not bucket_list:
        bucket_list = [(col, 10) for col in df.columns]

    for col, bin_amt in bucket_list:
        df[col] = pd.cut(
            df[col],
            bins=bin_amt,
            right=True,
            labels=range(1, bin_amt + 1),
        )
    return df


class ModelSelector():
    def __init__(self, acc_metric='accuracy_score', num_cv=5, **kwargs):
        '''
        kwargs: *something* must be passed for each type or it will be ignored.
        Can have just check, just ignore, just params, or any combination thereof.

        kwarg inputs:
            bucket: check, ignore, bucket_list.
            min_max_scale: check, ignore, feature_range.
            one_hot_encode: check, ignore, categories, bucket_list.
            raw: check, ignore.

        Example:
        kwargs = {
            'min_max_scale': {
                'ignore': ['RandForest'],
                'feature_range': (0, 0,5)
            },
            'one_hot_encode': {
                'check': ['GaussNB', 'MultiNB'],
                'categories': [
                    list(range(1, 11)) if c not in ['multiworld'] else [0, 1]
                    for c in df_train.columns
                ],
                'bucket_list': [(col, 10) for col in df.columns if col not in ['multiworld']]
            },
            raw: {
                'check': ['LogRegress']
            }
        }
        '''

        # Unpack the data preparation types and params.
        self.run_types = {}
        for k in kwargs:
            self.run_types[k] = kwargs.get(k, None)  # Should contain check, ignore, any params needed.

        self.acc_metric = acc_metric  # scoring param to optimize on in CV.
        self.num_cv = num_cv

        self.summary_df_cv = None  # best CV-search score.
        self.models = None  # dict of model objects.
        self.params = None  # dict of all best params for all evaluated models.
        self.best_model = None  # string, name of best model.
        self.best_params = None  # dict of best params for the best model.

    def fit(self, X_in, y_in, X_val_in=None, y_val_in=None):
        # get lists organized and initialize dicts.
        check_list = [
            'GMM', 'LogRegress', 'DecTree', 'RandForest',
            'SupportVC', 'kNN', 'GaussNB', 'MultiNB'
        ]  # default is all models.

        # Pull out models to check and run types, drop ignore models, default to all models.
        todo_list = [
            (mod, k)
            for k in self.run_types
            for mod in self.run_types[k].get('check', check_list)
            if self.run_types[k] and mod not in self.run_types[k].get('ignore', [])
        ]

        summary_dict_cv = {mod_tup: 0 for mod_tup in todo_list}  # eventually turn into df.
        model_dict = {mod_tup: 0 for mod_tup in todo_list}  # dict of model objects.
        params = {mod_tup: 0 for mod_tup in todo_list}  # stores params for each model.

        # loop over todo_list and score. Innefficient because re-prepping X.
        for model, prep_method in tqdm(todo_list, desc='Training models', ncols=150):
            t_0 = time.time()

            mod = globals()[model](acc_metric=self.acc_metric, num_cv=self.num_cv)  # Instantiate model class.

            # Prep data and fit model.
            X = self.data_prep(prep_method, X_in)
            if X_val_in:  # OoS score case.
                X_val = self.data_prep(prep_method, X_val_in)
                mod_score  = round(mod.score(X, y_in, X_val, y_val_in) * 100,  2)
            else:  # In-sample score case.
                mod_score = round(mod.fit(X, y_in).best_score * 100,  2)

            # Store results
            summary_dict_cv[(model, prep_method)] = {
                'time': time.strftime("%H:%M:%S",  time.gmtime(time.time()-t_0)),
                self.acc_metric: mod.best_score
            }

            params[(model, prep_method)] = mod.best_params
            model_dict[(model, prep_method)] = mod

        # get df and sort based on perf. store bests.
        summ_df_cv = pd.DataFrame.from_dict(summary_dict_cv, orient='index')
        summ_df_cv = summ_df_cv.sort_values(by=[self.acc_metric], ascending=False)

        self.best_model = summ_df_cv.index[0]
        self.best_params = params[self.best_model]
        self.params = params
        self.summary_df_cv = summ_df_cv
        self.models = model_dict

        return self

    def data_prep(self, prep_method, X_in):
        '''
        prep_method presumably gotten from MS.fit() loop or best_mod in external script.
        '''

        X = X_in.copy()  # Assumes already shuffled if needed.

        if prep_method == 'bucket':
            X = bucket_data(X, self.run_types[prep_method].get('bucket_list', None))
            X = X.values.astype(float)  # Tensorflow fails on ints.
        elif prep_method == 'min_max_scale':
            mms = MinMaxScaler(
                feature_range=self.run_types[prep_method].get('feature_range', (0, 1))
            )
            X = mms.fit_transform(X)
        elif prep_method == 'one_hot_encode':
            X = bucket_data(X, self.run_types[prep_method].get('bucket_list', None))
            enc = OneHotEncoder(
                categories=self.run_types[prep_method].get('categories', 'auto')
            )  # 'auto' is default argument for OHE, else take category list.
            X = enc.fit_transform(X).toarray()
        elif prep_method == 'raw':
            pass

        return X


class GaussNB():
    def __init__(self, acc_metric='accuracy_score', num_cv=5):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_params = None
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
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
    def __init__(self, acc_metric='accuracy_score', num_cv=5):
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_params = None
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
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
    def __init__(self, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
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
    def __init__(self, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0
        self.num_cv = num_cv

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
            clf = GridSearchCV(svc, n_jobs=CPU_USE, param_grid=parameters, scoring=self.acc_metric.split('_score')[0], cv=self.num_cv)
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
    def __init__(self, num_iter=200, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
            )

            try:
                clf.fit(X, Y)
            except ValueError:
                clf = GridSearchCV(
                    rf,
                    n_jobs=CPU_USE,
                    param_grid=parameters,
                    scoring=self.acc_metric.split('_score')[0],
                    cv=self.num_cv
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
    def __init__(self, num_iter=2500, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
            )

            try:
                clf.fit(X, Y)
            except ValueError:
                clf = GridSearchCV(
                    dc,
                    n_jobs=CPU_USE,
                    param_grid=parameters,
                    scoring=self.acc_metric.split('_score')[0],
                    cv=self.num_cv
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
    def __init__(self, num_iter=300, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.y_pred_prob = None
        self.label_prob = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.num_iter = num_iter
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
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
    def __init__(self, best_params=None, acc_metric='accuracy_score', num_cv=5):
        self.best_params = best_params
        self.y_pred = None
        self.acc_metric = acc_metric
        self.accuracy_score = None
        self.best_score = 0
        self.num_cv = num_cv

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
                cv=self.num_cv
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
