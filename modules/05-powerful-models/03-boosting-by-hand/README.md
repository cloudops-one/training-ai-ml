# 05.3 · Boosting: AdaBoost and gradient boosting by hand

⏱ about 2 hours · Prerequisites: 05.1, 05.2, 02.2–02.3 (residuals, SSE), 02.7 (tree regressor) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A random forest builds hundreds of trees **independently** and averages them. Every tree attacks the whole problem from scratch, and none of them learns from the others' mistakes.

That is wasteful. Imagine a team of analysts where the second analyst is told *"here are the customers the first one got wrong, focus on those"*, the third is told what the first two *together* still get wrong, and so on. Each analyst only needs to be a little bit useful, as long as they fix *something* the team was missing. That is **boosting**, and its modern versions (XGBoost, LightGBM, CatBoost) win most competitions on business tables.

This lesson builds boosting by hand on five customers you already know, so that "gradient boosting" stops being a magic phrase.

## 2. The idea

A golfer does not hit the ball into the hole in one stroke. The first shot gets close. The second shot is aimed not at the hole from the tee but at the **remaining distance**. The third fixes what is left after that. Each shot is small and imperfect, yet the sum of shots gets the ball in.

Gradient boosting does the same with predictions:

```
start:      predict the average for everyone              → big errors
tree 1:     learns the errors; add a fraction of it       → smaller errors
tree 2:     learns the errors that are still left         → smaller still
...
final prediction = start + ν·tree1 + ν·tree2 + ν·tree3 + …
```

Each tree is tiny, often a **stump** (one split, two leaves). The **learning rate** $\nu$ (nu) says how much of each tree's correction to trust. Small steps, many trees.

**AdaBoost** (1995), the first successful booster, used a different mechanism for the same idea: after each round it **raises the weight** of the rows it got wrong, so the next stump is forced to pay attention to them.

## 3. Definition

- **Boosting**: build models in sequence; each new model focuses on what the ensemble so far gets wrong; add them up.
- **Weak learner**: a model only slightly better than guessing, usually a shallow tree or stump.
- **Residual**: $r_i = y_i - F(x_i)$, what the current ensemble $F$ still misses for row $i$.
- **Gradient boosting (regression, squared error)**: each new tree is fitted to the **residuals**, then added with weight $\nu$.
- **Learning rate / shrinkage** $\nu$: between 0 and 1. Smaller = more cautious, needs more trees, usually generalises better.
- **AdaBoost**: each round fits a stump to the weighted data, gives it a vote $\alpha$ based on its weighted error, and increases the weights of misclassified rows.

Why "gradient"? For squared error, the residual $y - F(x)$ is exactly the downhill direction (negative gradient) of the loss with respect to the prediction. So each tree is a step of **gradient descent** (02.4), taken in the space of functions instead of the space of $m$ and $c$. For other losses (log-loss for churn), the tree is fitted to the gradient of that loss instead, but the recipe is the same.

## 4. Formula

**Gradient boosting** with $M$ rounds:

$$F_0(x) = \bar{y}$$
$$r_i^{(m)} = y_i - F_{m-1}(x_i) \qquad \text{(residuals)}$$
$$h_m = \text{small tree fitted to } (x_i, r_i^{(m)})$$
$$F_m(x) = F_{m-1}(x) + \nu \, h_m(x)$$

A stump $h_m$ predicts the **mean residual** in each of its two leaves, and chooses the split that minimises the sum of squared residuals around those means (the variance-reduction rule from 02.7).

**AdaBoost** for one round with weighted error $\varepsilon$:

$$\alpha = \tfrac{1}{2}\ln\!\left(\frac{1-\varepsilon}{\varepsilon}\right)$$

A stump with 20% weighted error gets $\alpha = \tfrac{1}{2}\ln 4 = 0.69$; one with 45% error gets only 0.10; one at 50% (a coin flip) gets 0. Misclassified rows have their weight multiplied by $e^{\alpha}$, correct ones by $e^{-\alpha}$, then weights are renormalised.

## 5. By hand

The five customers from 02.2. Learning rate $\nu = 0.5$, stumps.

| customer | visits $x$ | spend $y$ |
|---|---|---|
| C001 | 2 | 50 |
| C002 | 4 | 80 |
| C003 | 6 | 90 |
| C004 | 8 | 130 |
| C005 | 10 | 150 |

**Round 0: start from the mean.** $F_0 = 100$ for everyone.

Residuals: $-50, -20, -10, +30, +50$. Sum of squared residuals (SSE) $= 2500 + 400 + 100 + 900 + 2500 = \mathbf{6400}$.

**Round 1: fit a stump to the residuals.** Try each split between neighbouring $x$ values. For each, the left leaf predicts the mean residual on its side, the right leaf likewise.

| split | left residuals → mean | right residuals → mean | SSE around leaf means |
|---|---|---|---|
| $x \le 3$ | −50 → −50 | −20, −10, 30, 50 → 12.5 | 3275.0 |
| $x \le 5$ | −50, −20 → −35 | −10, 30, 50 → 23.33 | 2316.7 |
| $x \le 7$ | −50, −20, −10 → **−26.67** | 30, 50 → **40** | **1066.7** ✓ |
| $x \le 9$ | −50, −20, −10, 30 → −12.5 | 50 → 50 | 3275.0 |

Best: split at $x \le 7$. Add half of it ($\nu = 0.5$):

- customers with $x \le 7$: $100 + 0.5 \times (-26.67) = 86.67$
- customers with $x > 7$: $100 + 0.5 \times 40 = 120$

| $x$ | $y$ | $F_1$ | residual |
|---|---|---|---|
| 2 | 50 | 86.67 | −36.67 |
| 4 | 80 | 86.67 | −6.67 |
| 6 | 90 | 86.67 | 3.33 |
| 8 | 130 | 120 | 10 |
| 10 | 150 | 120 | 30 |

SSE: 6400 → **2400**.

**Round 2.** Repeat on the new residuals. Best split is now $x \le 3$ (left mean −36.67, right mean 9.17, SSE 719.4). Add half:

$F_2 = 68.33,\ 91.25,\ 91.25,\ 124.58,\ 124.58$. SSE → **1139.6**.

**Round 3.** Best split $x \le 9$ (left mean −6.35, right mean 25.42). Add half:

$F_3 = 65.16,\ 88.07,\ 88.07,\ 121.41,\ 137.29$. SSE → **534.0**.

The error fell 6400 → 2400 → 1140 → 534 using three one-question trees. scikit-learn's `GradientBoostingRegressor(n_estimators=3, learning_rate=0.5, max_depth=1)` gives exactly the same $F_3$, to the penny.

Notice that each round picked a **different** split: the second and third trees fixed problems the first one could not see. And no single stump looks anything like the true relationship; only their sum does.

## 6. In code

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

gbr = GradientBoostingRegressor(n_estimators=3, learning_rate=0.5, max_depth=1)
gbr.fit(X, y)                     # same F_3 as the hand calculation

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbc.fit(X_train, y_train)          # for churn: trees fit the gradient of log-loss
```

The notebook also writes the boosting loop itself in about ten lines of NumPy, and prints `staged_predict` from scikit-learn so you can watch the prediction move round by round.

## 7. On real data

On the churn table, `GradientBoostingClassifier` with default settings (100 trees, depth 3, learning rate 0.1) scores CV AUC **0.730**, test AUC **0.727**: in the same range as the forest and slightly below the logistic benchmark (0.740 / 0.741).

The notebook then plots **test AUC after each boosting round** for three learning rates:

| learning rate | peak test AUC | at round | after 500 rounds |
|---|---|---|---|
| 0.3 | 0.739 | 6 | 0.666 |
| 0.1 | 0.740 | 15 | 0.704 |
| 0.03 | 0.741 | 51 | 0.724 |

All three reach about the same peak, level with logistic regression. The difference is what happens *after* the peak: the large learning rate collapses as later trees fit noise, the small one declines slowly. It also explains the default model's 0.727: 100 rounds at learning rate 0.1 is already well past that setting's best round (15). Picking the round by watching the *test* set is cheating, of course; 05.4 shows how to stop at the right round automatically with a validation set.

## 8. Where it breaks

- **Boosting can overfit.** Unlike a forest, adding more boosting rounds *does* eventually hurt: later trees chase noise. Control it with the learning rate, tree size, subsampling, and early stopping.
- **Sequential = slower to train** than a forest of the same size, because round $m$ needs round $m-1$. Modern libraries parallelise *inside* each tree to compensate.
- **Sensitive to outliers** with squared-error loss: a huge residual dominates the next tree. Use a robust loss (`loss="huber"`) for messy targets.
- **AdaBoost is sensitive to label noise**: mislabelled rows get ever-larger weights. Gradient boosting with log-loss is more forgiving, which is why it replaced AdaBoost in practice.
- **Many knobs.** Learning rate, number of trees, depth and subsampling interact; tune them together (05.6).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: redo round 1 with $\nu = 1$ (no shrinkage). What are $F_1$ and the SSE? Is the bigger step better after one round? After three (use code)?
2. Write the boosting loop in NumPy for 50 rounds with $\nu = 0.5$ and plot SSE against round. Does it reach zero? Why is reaching zero not the goal?
3. AdaBoost by hand: a stump has weighted error 0.3. Compute $\alpha$ and the factor by which a misclassified row's weight is multiplied.
4. On churn, plot staged test AUC for `learning_rate` 0.3, 0.1, 0.03 up to 500 trees. For each, at which round does test AUC peak?

**Quiz**

1. In boosting, what does each new tree try to predict?
2. What does the learning rate do, and what do you trade by making it smaller?
3. Why is it called *gradient* boosting?
4. More trees can never hurt a random forest. Is the same true for boosting?
5. How does AdaBoost make the next stump focus on hard rows?

## 10. Recap

- **Boosting** adds small models one at a time, each correcting what the sum so far still gets wrong.
- **Gradient boosting** starts at the mean, fits each tree to the **residuals** (the negative gradient), and adds it scaled by a **learning rate**.
- By hand on five customers: SSE 6400 → 2400 → 1140 → 534 in three stumps, matching scikit-learn exactly.
- **AdaBoost** does the same job by re-weighting misclassified rows and giving each stump a vote $\alpha = \tfrac12 \ln\frac{1-\varepsilon}{\varepsilon}$.
- Boosting reduces bias but can overfit with too many rounds; learning rate and early stopping control it.

**Next:** [05.4 XGBoost, LightGBM and CatBoost](../04-xgboost-lightgbm/): the industrial-strength versions, and when they actually win.
