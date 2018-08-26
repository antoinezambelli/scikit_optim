# scikit-optim
Convenience Library for CV grid-search optimization of scikit classifiers.

## Description

This is a convenience library for use with `scikit` supervised learning algorithms (`knn`,`Logistic Regression`, etc). It provides methods to automatically conduct a cross-validated grid-search of hyperparameter tuning - so you odn't need to code loops every time.

It also includes a `ModelSelector` which will run comparisons of all models - returning runtimes and accuracy scores for different models.

## Usage

Assume the user will split their dataset into training and test sets - `X_tr`/`Y_tr` and `X_te`/`Y_te`, respectively.

### Example Prediction of known data
```
logreg = LogRegress()
Y_te_pred = logreg.fit(X_tr, Y_tr).predict(X_tr, Y_tr, X_te)
```
We could then compare the accuracy of `Y_te_pred` to the known `Y_te`. Alternatively, we cando it automatically
```
logreg = LogRegress()
accu = logreg.score(X_tr, Y_tr, X_te, Y_te)  # compares Y_te (pedicted) to Y_te (actual) automatically
```

### Example prediction of unkown data

Using the full known dataset `X`/`Y` and unknown dataset `X_un`, just call
```
logreg = LogRegress()
Y_un_pred = logreg.fit(X, Y).predict(X, Y, X_un)
```

### ModelSelector

The `ModelSelector` provides an easy way to compare preliminary (though optimized) results on different models.

Calling
```
model_sel = ModelSelector()
model_sel.fit(X_tr, Y_tr, X_te, Y_te)
```
will return the (ranked) accuracy scores of the different models and the computation times required. Can be accessed via the `pandas` dataframe
```
model_sel.summary_df
```

#### Parameters

By default `ModelSelector` will check every model. We can pass a list of model names (strings) to either `check` or `ignore`.

To only run and compare `kNN` and `LogRegress` models:
```
model_sel = ModelSelector(check=['kNN','LogRegress'])
```
To check every model *except* `kNN` and `LogRegress`:
```
model_sel = ModelSelector(ignore=['kNN','LogRegress'])
```

## Supported Models

For now, we support the following model names. Equivalent `scikit` models can be found in the `import` section of the source code.
```
['GMM', 'LogRegress', 'DecTree', 'RandForest', 'SupportVC', 'kNN', 'BGMM', 'GaussNB', 'MultiNB']
```
Note that `GMM` and `BGMM` are kernel density estimation classifier that are not in `scikit`. They fit a `GaussianMixture` or `BayesianGaussianMixture` to each label, and use the probability of an unknown point being sampled from either distribution to classify it.

