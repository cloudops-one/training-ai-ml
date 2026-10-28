# scikit-learn

Every estimator has the same shape. Learn the pattern once ([02.2](../modules/02-regression/02-linear-regression-by-hand/)), then everything else is a different class name.

| Method | Does | Used by |
|---|---|---|
| `fit(X, y)` | learn from training data (`fit(X)` for unsupervised) | every model and transformer |
| `predict(X)` | labels or numbers | models |
| `predict_proba(X)` | class probabilities, shape (rows, classes) | classifiers |
| `transform(X)` | change the data with what `fit` learned | scalers, encoders, PCA |
| `fit_transform(X)` | both, **on training data only** | transformers |
| `score(X, y)` | accuracy (classifiers) or R² (regressors) | models |

`X` is 2-D (rows × features), `y` is 1-D. One feature → `X = df[["visits"]]` (double brackets), not `df["visits"]`.

## Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)       # stratify keeps the churn rate

# Time split: the future must stay in the test set
cutoff = "2011-09-01"
train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
```

→ [04.1](../modules/04-trustworthy-models/01-train-validation-test/), leakage traps in [04.2](../modules/04-trustworthy-models/02-data-leakage/)

## Pipeline + ColumnTransformer (the leak-proof template)

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = ["orders", "spend", "days_since_last"]
categorical = ["country"]

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), categorical),
])
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)                  # imputer and scaler learn from train only
p = model.predict_proba(X_test)[:, 1]        # raw new rows in, probabilities out
```

Inspect what the preprocessing produced: `preprocess.set_output(transform="pandas")`. → [04.8](../modules/04-trustworthy-models/08-pipelines/)

## Cross-validation

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"AUC {scores.mean():.3f} ± {scores.std():.3f}")

res = cross_validate(model, X_train, y_train, cv=cv,
                     scoring=["roc_auc", "f1"], return_train_score=True)
ts_cv = TimeSeriesSplit(n_splits=5)          # for time-ordered rows
```

Pass the **whole pipeline** to CV so preprocessing is refitted inside every fold. → [04.7](../modules/04-trustworthy-models/07-cross-validation/)

## Tuning

```python
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

grid = GridSearchCV(model, {"clf__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, round(grid.best_score_, 3))

rand = RandomizedSearchCV(model, {"clf__C": loguniform(1e-3, 1e2)}, n_iter=20,
                          cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1)
```

Parameter names are `step__param` (`clf__C`, `prep__num__simpleimputer__strategy`). Score the test set **once**, after choosing. → [05.6](../modules/05-powerful-models/06-hyperparameter-tuning/)

## Models at a glance

| Task | Import |
|---|---|
| Linear / regularised | `from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression` |
| Neighbours | `from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor` |
| Trees and ensembles | `from sklearn.tree import DecisionTreeClassifier` · `from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier` |
| Boosting libraries | `from xgboost import XGBClassifier` · `from lightgbm import LGBMClassifier` (`verbose=-1`) |
| SVM / Naive Bayes | `from sklearn.svm import SVC` · `from sklearn.naive_bayes import GaussianNB, MultinomialNB` |
| Clustering | `from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering` · `from sklearn.mixture import GaussianMixture` |
| Dimensions / anomalies | `from sklearn.decomposition import PCA` · `from sklearn.ensemble import IsolationForest` |
| Text | `from sklearn.feature_extraction.text import TfidfVectorizer` |

## Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, model.predict(X_test)))
```

Full list with formulas: [metrics.md](metrics.md).

## Explain

```python
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0)
```

Prefer permutation importance over `feature_importances_` (*a random noise column got impurity importance 0.175 but permutation 0.001*). SHAP for single predictions. → [05.8](../modules/05-powerful-models/08-explaining-models/)

## Save and load

```python
import joblib
joblib.dump(model, "models/churn_logreg_2026-10-28_auc0.741.joblib")   # the whole pipeline
model = joblib.load("models/churn_logreg_2026-10-28_auc0.741.joblib")
```

Load with the same scikit-learn version you saved with. → [05.10](../modules/05-powerful-models/10-choosing-and-saving/)

## Common errors

| Error / symptom | Cause | Fix |
|---|---|---|
| `Expected 2D array, got 1D array` | `X` is a Series or 1-D array | `df[["col"]]` or `x.reshape(-1, 1)` |
| `could not convert string to float` | Text column reached the model | One-hot encode it in a ColumnTransformer |
| `Input X contains NaN` | Missing values | `SimpleImputer` (or a model that accepts NaN: HistGradientBoosting, XGBoost, LightGBM) |
| `Found unknown categories` | New category in test data | `OneHotEncoder(handle_unknown="ignore")` |
| `ConvergenceWarning` (logistic) | Unscaled features or too few iterations | `StandardScaler`, `max_iter=1000` |
| Test score far above what live data gives | Leakage (post-cut-off feature, preprocessing before split, duplicates) | Pipeline + time split; audit features ([04.2](../modules/04-trustworthy-models/02-data-leakage/)) |
| `roc_auc_score` looks like accuracy | Passed labels, not scores | Use `predict_proba(X)[:, 1]` |
| Train 1.00, test much lower | Overfitting | Limit depth, `min_samples_leaf`, regularise ([04.9](../modules/04-trustworthy-models/09-bias-variance/)) |
