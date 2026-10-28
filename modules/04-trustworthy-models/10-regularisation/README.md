# 04.10 · Regularisation: Ridge and Lasso

⏱ about 1 hour · Prerequisites: 02.2 (least squares), 02.5 (multiple regression), 04.9 (bias and variance) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The finance team sends you a spreadsheet of 20 customers with 11 columns and asks for a spend model. Three columns are clearly meaningful (visits, items, days since last order). The other eight are things like "survey score batch 3" and "promo code hash", and nobody knows whether they matter.

You fit ordinary least squares. It happily gives every column a coefficient, some of the meaningless ones as large as the meaningful ones. Its cross-validated error is £14.71; a model told which three columns are real gets £10.29. With 20 rows and 11 columns, least squares has enough freedom to fit noise, which is the high-variance problem from 04.9.

You want a model that, without being told, **prefers to ignore columns that do not earn their place**.

## 2. The idea

Think of a household budget. Without any rule, you will find a reason to spend a little on everything. Add a rule, "every purchase must justify its cost", and the marginal purchases disappear while the essentials stay.

**Regularisation** adds that rule to training. The loss becomes:

```
   how badly the model fits the data   +   α × how big the coefficients are
   ─────────────────────────────────       ──────────────────────────────────
        (the usual squared error)               (the cost of complexity)
```

A coefficient now has to reduce the error by more than it costs. Useful features clear the bar; noise features do not. The knob **α** (alpha) sets the price: α = 0 is ordinary least squares, very large α makes every coefficient too expensive and the model predicts the mean.

## 3. Definition

- **Regularisation**: adding a penalty on coefficient size to the training loss, to reduce variance (overfitting) at the cost of a little bias.
- **Ridge regression (L2)**: penalty = sum of **squared** coefficients. Shrinks all coefficients smoothly towards zero; never exactly zero.
- **Lasso regression (L1)**: penalty = sum of **absolute** coefficients. Can set coefficients **exactly to zero**, so it also selects features.
- **ElasticNet**: a mix of both, controlled by `l1_ratio`.
- **α (alpha)**: penalty strength, chosen by cross-validation. In `LogisticRegression` the knob is **C = 1/α** (small C = strong penalty), and an L2 penalty is on **by default**.
- The **intercept is not penalised**; only the slopes are.

## 4. Formula

With predictions $\hat{y}_i = c + \sum_j \beta_j x_{ij}$:

**Ridge** (as scikit-learn writes it):

$$\text{loss} = \sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j \beta_j^2$$

**Lasso** (scikit-learn divides the error part by $2n$):

$$\text{loss} = \frac{1}{2n}\sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j |\beta_j|$$

For **one feature** both have exact solutions, using $S_{xy} = \sum (x - \bar{x})(y - \bar{y})$ and $S_{xx} = \sum (x - \bar{x})^2$ from 02.2:

$$m_{\text{ridge}} = \frac{S_{xy}}{S_{xx} + \alpha}
\qquad
m_{\text{lasso}} = \max\!\left(0,\ \frac{S_{xy}/n - \alpha}{S_{xx}/n}\right)
\qquad
c = \bar{y} - m\,\bar{x}$$

Ridge adds α to the denominator: the slope shrinks but never reaches zero. Lasso subtracts α from the numerator: once α reaches $S_{xy}/n$ the slope is **exactly** zero.

## 5. By hand

The five customers from 02.2: visits $x = [2, 4, 6, 8, 10]$, spend $y = [50, 80, 90, 130, 150]$. Known: $\bar{x} = 6$, $\bar{y} = 100$, $S_{xy} = 500$, $S_{xx} = 40$, $n = 5$. Least squares: $m = 12.5$, $c = 25$, SSE = 150.

**Step 1: compare two lines without a penalty.** Line P = (12.5, 25), SSE = 150. Line Q = (12, 28): predictions 52, 76, 100, 124, 148; residuals −2, 4, −10, 6, 2; squares 4, 16, 100, 36, 4; SSE = **160**. P wins.

**Step 2: add a Ridge penalty, α = 1.**

- P: 150 + 1 × 12.5² = 150 + 156.25 = **306.25**
- Q: 160 + 1 × 12² = 160 + 144 = **304.00**

Q now wins. The penalty made the flatter line worth a slightly worse fit.

**Step 3: the exact Ridge answer for α = 1.** $m = 500 / (40 + 1) = 12.195$, $c = 100 - 12.195 \times 6 = 26.83$. For α = 10: $m = 500/50 = 10$, $c = 40$. For α = 40: $m = 500/80 = 6.25$, $c = 62.5$.

**Step 4: Lasso, α = 20.** Loss = SSE / 10 + 20 × |m|.

- P: 150/10 + 20 × 12.5 = 15 + 250 = **265**
- Q: 160/10 + 20 × 12 = 16 + 240 = **256**

Exact answer: $m = (500/5 - 20)/(40/5) = (100 - 20)/8 = 10$, $c = 40$.

**Step 5: where Lasso hits zero.** $m = (100 - \alpha)/8$ is zero at **α = 100**. Then $c = \bar{y} = 100$: the model predicts the mean for everyone. Ridge at α = 100 still has $m = 500/140 = 3.57$.

| α | Ridge m | Lasso m |
|---|---|---|
| 0 | 12.5 | 12.5 |
| 1 | 12.195 | 12.375 |
| 10 / 20 | 10 (α=10) | 10 (α=20) |
| 40 / 50 | 6.25 (α=40) | 6.25 (α=50) |
| 100 | 3.57 | **0** |

## 6. In code

```python
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

Ridge(alpha=1).fit(X5, y).coef_            # [12.195]  matches the formula
Lasso(alpha=20).fit(X5, y).coef_           # [10.0]

# in practice: scale first, choose alpha by cross-validation
model = make_pipeline(StandardScaler(), LassoCV(cv=5))
```

`LogisticRegression(penalty="l1", C=0.01, solver="liblinear")` is Lasso for classification.

## 7. On real data

**Twenty customers, three real features plus eight pure-noise columns** (5-fold CV mean absolute error):

| Model | CV MAE |
|---|---|
| least squares, all 11 columns | £14.71 |
| RidgeCV, all 11 | £15.41 |
| **LassoCV, all 11** | **£12.63** |
| least squares, told the 3 real columns | £10.29 |

Lasso kept all three real features and set **seven of the eight noise columns to exactly zero**. Ridge shrank everything but kept every noise column, and did not help. ElasticNetCV chose pure Lasso (`l1_ratio = 1.0`).

**Churn, ten engineered features, L1 logistic regression** (5-fold CV AUC):

| C | features kept | CV AUC |
|---|---|---|
| 0.001 | 0 | 0.500 |
| 0.003 | 1 | 0.722 |
| **0.01** | **5** | **0.741** |
| 0.1 | 8 | 0.742 |
| 10 | 10 | 0.742 |

Five features (orders, spend, products, days since last order, orders in the last three months) do as well as all ten.

## 8. Where it breaks

- **Scale first, always.** The penalty is on coefficient size, and coefficient size depends on units. Unscaled, a feature measured in days and one measured in items are penalised unequally for the same usefulness.
- **Lasso picks one of a correlated group.** If `spend` and `log_spend` are both present, Lasso tends to keep one arbitrarily. ElasticNet keeps the group. Do not read "Lasso dropped it" as "it does not matter".
- **Ridge does not select.** When most features are useless, Ridge alone will not save you (exercise 3: £37 → £38 with 30 noise columns; Lasso £15).
- **CV does not reward simplicity by itself.** It picks the best score, even if a much smaller model is within 0.001. Choose the simpler one deliberately if it matters.
- **Too much penalty underfits.** At C = 0.001 the churn model ignores every feature (AUC 0.5). Regularisation trades variance for bias; it does not remove the trade-off.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: Ridge slope and intercept on the five-row table for α = 20 and α = 160. What does the line become as α → ∞?
2. Fit LassoCV on the noisy 20-customer table without scaling and compare the coefficients.
3. Use 30 noise columns instead of 8 and compare least squares, Ridge and Lasso.
4. Choose C for L1 logistic regression on churn with `LogisticRegressionCV`, then score once on the test set.

**Quiz**

1. What is regularisation, in one sentence, and which of bias or variance does it reduce?
2. How do Ridge and Lasso differ in what they do to coefficients?
3. Ridge on the five-row table with α = 10: what are m and c?
4. Why must you scale features before Ridge or Lasso?
5. In `LogisticRegression`, does a stronger penalty mean a larger or smaller C?

## 10. Recap

- **Regularisation** = error + α × coefficient size. It trades a little bias for less variance.
- **Ridge (L2)** shrinks smoothly: $m = S_{xy}/(S_{xx} + \alpha)$. **Lasso (L1)** shrinks to exactly zero and selects features.
- Choose α (or C = 1/α) by **cross-validation**, and **scale first**, inside a pipeline.
- On ShopWave, Lasso removed noise columns that least squares and Ridge kept, and five churn features matched ten.
- `LogisticRegression` is regularised by default; now you know what `C` means.

**Next:** [04.P Mini-project: rebuild churn as a leak-free pipeline](../project/) puts every lesson of this module into one honest workflow.
