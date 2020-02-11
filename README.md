# scikit-optim
Convenience Library for CV grid-search optimization of scikit classifiers.

## Description

This is a convenience library for use with `scikit` supervised learning algorithms (`knn`,`Logistic Regression`, etc). It provides methods to automatically conduct a cross-validated grid-search for hyperparameter tuning - so you don't need to code loops every time.

It also includes a `ModelSelector` which will run comparisons of all models - returning runtimes and accuracy scores for different models.

It also includes data prep functionality (minmax scaling, bucketing, normalizing) and will compare those methods as well.

## Usage
Typical usage of `ModelSelector`:

```
ms_params = {
    'bucket': {
        'check': CHECK,
        'bucket_list': [(col, 10) for col in df_train.columns if col not in ['my_col']]
    },
    'min_max_scale': {
        'check': ['MultiNB', 'GaussNB', 'DecTree']
    },
    'one_hot_encode': {
        'check': CHECK,
        'categories': [
            list(range(1, 11)) if c not in ['my_col'] else [0, 1]
            for c in df_train.columns
        ],
        'bucket_list': [(col, 10) for col in df_train.columns if col not in ['my_col']]
    }
}  # kwargs for scikit_optim v4.0.

# Fit MS with params on df_train, y_train.
model_sel = ModelSelector(acc_metric='precision_score', num_cv=5, **ms_params)
model_sel.fit(df_train, y_train)

# Prep the data sets according to best performing model + data prep combo.
X_train = model_sel.data_prep(
    model_sel.summary_df_cv.index[0][1],
    df_train
)
X_test = model_sel.data_prep(
    model_sel.summary_df_cv.index[0][1],
    df_test
)

# Predict X_test using the fitted best model.
best_mod = model_sel.models[model_sel.summary_df_cv.index[0]]
res = {
    'y_pred': best_mod.predict(
        X_train,
        y_train,
        X_test,
    ),
    'y_pred_prob': best_mod.predict_proba(
        X_train,
        y_train,
        X_test,
    )
}
```

Note that if no validation sets are explicitly passed in, then the library will use training performance to rank models (as in the example above).



DOCS BELOW THIS POINT ARE PARTIALLY OUTDATED (but can still give an idea of what the library does).
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
will return the several important elements:

#### Summary DataFrame
(ranked) accuracy scores of the different models and the computation times required. Can be accessed via the `pandas` dataframe
```
model_sel.summary_df
```

#### Best Model
The best model (instance) is also returned, with the best parameters it found in training.
```
best_mod = model_sel.models[model_sel.best_model]
```
Note that running
```
Y_un_pred = best_mod.fit(X,Y).predict(X,Y,X_un)
```
or
```
best_mod.best_params = None  # this line would reset the params, and fit() would re-optimize for the full set.
Y_un_pred = best_mod.fit(X,Y).predict(X,Y,X_un)
```
will yield different results. In the latter case, the prediction will be done with parameters optimized over all of `X,Y`. In the former case, we would use parameters optimized over `X_tr,Y_tr`. The latter requires recomputation but uses more of the data for tuning.

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

