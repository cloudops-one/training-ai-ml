# 05.6 · Hyperparameter tuning: grid, random, Bayesian

⏱ about 1.5 hours · Prerequisites: 04.7 (cross-validation), 05.4 (LightGBM) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You swapped the churn model from logistic regression to LightGBM, the algorithm that wins most tabular competitions. Its test AUC went **down**: 0.693, against 0.741 for logistic regression.

Nothing is broken. LightGBM ships with default settings (31 leaves per tree, learning rate 0.1) designed for datasets with hundreds of thousands of rows. ShopWave's churn table has 2,487 training customers. With those defaults, every tree is too big and the model memorises noise.

These settings are called **hyperparameters**, and choosing them well is called **tuning**. The question for this lesson: *how do you pick them without fooling yourself, and how much does it actually help?*

## 2. The idea

Tuning is a search. You have a few knobs; each combination of knob positions gives a model; you want the combination that scores best on customers the model has not seen.

Three ways to search:

- **Grid search**: list a few values per knob and try every combination. Simple, thorough, wasteful.
- **Random search**: draw combinations at random from ranges you give. With the same budget it tries more *distinct* values of each knob.
- **Bayesian search** (Optuna): look at the scores so far, guess where good settings are likely, and try there next. Like a treasure hunter who digs near the spots where they already found coins.

All three score each candidate with **cross-validation on the training set** (Module 04). The test set stays locked until the very end, because if you choose settings by their test score, the test score stops being an honest estimate.

## 3. Definition

- **Parameter**: a number the model learns during `fit` (a split threshold, a coefficient).
- **Hyperparameter**: a setting you choose before `fit` that controls *how* the model learns (`num_leaves`, `learning_rate`, `min_child_samples`, `max_depth`).
- **Search space**: the values or ranges you allow for each hyperparameter.
- **Budget**: how many settings you are willing to try.
- **Grid search** (`GridSearchCV`): try every combination of the listed values.
- **Random search** (`RandomizedSearchCV`): try `n_iter` combinations sampled from distributions.
- **Bayesian optimisation** (Optuna's TPE sampler): choose each new setting using a model of how earlier settings scored.

The three LightGBM knobs used here:

| Hyperparameter | What it controls | Small value | Large value |
|---|---|---|---|
| `num_leaves` | how many leaves each tree may have | simple trees, may underfit | complex trees, may overfit |
| `learning_rate` | how much each new tree corrects the previous ones | slow, careful learning | fast, jumpy learning |
| `min_child_samples` | fewest customers allowed in a leaf | leaves fit tiny groups (noise) | leaves only for big, reliable groups |

## 4. Formula

**Cost of a search** (the thing you can compute before pressing run):

$$\text{fits} = (\text{number of settings}) \times k \; + \; 1$$

where $k$ is the number of CV folds and the $+1$ is the final refit on all training data.

For a grid, the number of settings is the product of the list lengths:

$$\text{settings}_{\text{grid}} = n_1 \times n_2 \times \dots \times n_p$$

For random and Bayesian search, it is whatever budget you choose (`n_iter`, `n_trials`).

**Score of a setting** is the mean of its fold scores, and the spread tells you how much to trust differences:

$$\text{CV score} = \frac{1}{k}\sum_{i=1}^{k} \text{AUC}_i \qquad \text{noise} \approx \text{std}(\text{AUC}_1, \dots, \text{AUC}_k)$$

If two settings differ by less than the fold std, treat them as a tie.

## 5. By hand

**a) Counting fits.** The grid in the notebook is

```
learning_rate ∈ {0.01, 0.03, 0.1}       → 3 values
num_leaves    ∈ {4, 8, 31}              → 3 values
5-fold CV
```

Settings = 3 × 3 = **9**. Fits = 9 × 5 + 1 = **46**.

Add `min_child_samples ∈ {10, 40, 80}` and it becomes 27 settings and 136 fits. Every new knob in a grid **multiplies** the cost. Five knobs with five values each would be 3,125 settings, 15,626 fits. That is why grids stop at two or three knobs.

**b) Grid vs random with 9 tries.** Suppose only the learning rate matters. Sketch a square: learning rate along the bottom, leaves up the side.

```
Grid (9 tries)                Random (9 tries)
num_leaves                    num_leaves
  |  x     x     x              |    x         x
  |                             |  x      x
  |  x     x     x              |        x    x
  |                             |    x      x
  |  x     x     x              |               x
  +------------------ lr        +------------------ lr
     only 3 lr values              9 different lr values
```

The grid tests the same three learning rates three times each; random search tests nine. In the notebook's toy example the grid's best is 0.729 and random's is 0.750 (the true maximum).

**c) Is the gain real?** The default LightGBM's five fold AUCs are 0.681, 0.738, 0.658, 0.687, 0.726.

Mean = (0.681 + 0.738 + 0.658 + 0.687 + 0.726) / 5 = 3.490 / 5 = **0.698**.

Std ≈ **0.030**. So a tuned setting at 0.744 is about 0.046 better: more than the noise, a real gain. But two tuned settings at 0.742 and 0.745 differ by 0.003: a tie.

## 6. In code

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, randint

grid = GridSearchCV(model, {"learning_rate": [0.01, 0.03, 0.1], "num_leaves": [4, 8, 31]},
                    cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)

rand = RandomizedSearchCV(model, {"learning_rate": loguniform(0.005, 0.3),
                                  "num_leaves": randint(2, 64),
                                  "min_child_samples": randint(5, 100)},
                          n_iter=20, cv=cv, scoring="roc_auc", random_state=42).fit(X_train, y_train)

grid.best_params_, grid.best_score_, grid.best_estimator_
```

Optuna in ten lines:

```python
def objective(trial):
    model = lgb.LGBMClassifier(
        n_estimators=200, verbose=-1,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100))
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
study.best_params
```

`loguniform` spreads draws evenly across orders of magnitude, so 0.005–0.05 gets as many tries as 0.05–0.3. Use it for learning rates and regularisation strengths.

## 7. On real data

All searches run on the 2,487 training customers. Results:

| Model | CV AUC (train) | Test AUC (opened once) |
|---|---|---|
| Logistic regression (Module 03) | | 0.741 |
| LightGBM, defaults | 0.698 | 0.693 |
| LightGBM, grid (lr 0.01, 4 leaves) | 0.744 | 0.739 |
| LightGBM, random search (20 tries) | 0.742 | 0.734 |
| LightGBM, Optuna (30 trials) | 0.745 | 0.740 |

The grid's heat-map in the notebook tells the story: small trees and slow learning win; the default corner (31 leaves, 0.1) is worst.

Two conclusions, both important:

1. **Tuning rescued LightGBM**, lifting it by about 0.045 AUC, well above the fold noise of ±0.03. Bad defaults are the most common reason a "powerful" model disappoints.
2. **Tuning did not make it beat logistic regression.** Five features and 2,500 customers do not contain much non-linear structure. The three tuned versions are tied with each other and with the baseline, which matches what 05.2–05.4 found for every ensemble on this table. (On the fraud data in 05.4, where the pattern really is non-linear, trees win clearly.) Better features (Module 04.6) would buy more than any further search.

## 8. Where it breaks

- **Tuning on the test set.** The most common silent mistake. Choose by CV on training data; open the test set once.
- **Over-tuning to the CV folds.** Try thousands of settings and some will score well by luck on those particular folds. Keep budgets modest and treat ties as ties.
- **Grids explode.** Each new knob multiplies the cost. Use random or Bayesian search beyond two or three knobs.
- **Searching the wrong range.** If the best value sits on the edge of your range (e.g. `num_leaves=4` when you allowed 4–31), widen the range and search again.
- **Leakage inside the search.** If preprocessing (scaling, imputation, encoding) happens before the search instead of inside a Pipeline, every fold sees information from the others. Always search over a Pipeline (Module 04.8).
- **Thread fights.** Running a multi-threaded model inside a parallel search makes both slow. Set the model to `n_jobs=1` and let the search use `n_jobs=-1`.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: count settings and fits for a three-knob grid, and estimate run time on 1 and 8 cores.
2. Add `min_child_samples` to the grid. How many fits? Does the best score move by more than the noise?
3. Tune a random forest with `RandomizedSearchCV`. How does it compare with tuned LightGBM?
4. Run Optuna with three different seeds. Do the best settings agree? Do the best scores?

**Quiz**

1. What is the difference between a parameter and a hyperparameter? Give one of each for a decision tree.
2. A grid has 4 values of one knob, 3 of another, 2 of a third, with 5-fold CV. How many fits?
3. Why must you not choose hyperparameters by their test-set score?
4. With the same budget, why does random search usually beat grid search?
5. What does Optuna do that random search does not?

## 10. Recap

- Hyperparameters control how a model learns; defaults are rarely right for your data size.
- Score every candidate with **cross-validation on training data**; open the test set once at the end.
- Grid search multiplies cost with every knob; **random search** covers more values per knob for the same budget; **Optuna** learns where to look.
- Differences smaller than the fold std are ties.
- Tuning fixes bad defaults and buys a little more. Features and data buy far more.

**Next:** [05.7 Imbalanced data](../07-imbalanced-data/) handles the case where the thing you want to catch is rare.
