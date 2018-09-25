"""
#
# scikit_optim_kaggle_example.py
#
# Copyright (c) 2018 Antoine Emil Zambelli. MIT License.
#
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from scikit_optim import ModelSelector

train_df = pd.read_csv('cleaned_train.csv',index_col=0)
test_df = pd.read_csv('cleaned_test.csv',index_col=0)

X = train_df.drop("Survived", axis=1)  # Known training data.
Y = train_df["Survived"]
X_un  = test_df.drop("PassengerId", axis=1).copy()  # Unknown data.

# Get in-sample and out-of-sample slices. Here called X_tr and X_te.
m = int(0.6*len(X))
X_te = X.iloc[m:]
X_te = X_te.reset_index(drop=True)
Y_te = Y.iloc[m:]
Y_te = Y_te.reset_index(drop=True)

X_tr = X[0:m]
Y_tr = Y[0:m]

# We now have X, Y, X_tr, Y_tr, X_te, Y_te as in the readme. Run ModelSelector() with kNN and LogRegress.
model_sel = ModelSelector(check=['kNN','LogRegress'])
model_sel.fit(X_tr, Y_tr, X_te, Y_te)
# print(model_sel.summary_df)  # could manually inspect accuracy+runtimes to pick a preferred model.

# Take the best model as found by ModelSelector(), and execute a fit-predict on the full set.
best_mod = model_sel.models[model_sel.best_model]  # best model object, contains best params.
# best_mod.best_params = None  # this line would reset the params, and fit() would re-optimize for the full set.
Y_un_pred = best_mod.fit(X,Y).predict(X,Y,X_un)
