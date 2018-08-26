import numpy as np
import pandas as pd

from scikit_optim.scikit_optim import ModelSelector

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
model_sel = ModelSelector(check=['kNN','LogRegress'],verbose=True)
model_sel.fit(X_tr, Y_tr, X_te, Y_te)
print(model_sel.summary_df)

