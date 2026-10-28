# 02.3 · Measuring error: MAE, MSE, RMSE, R²

⏱ about 1.5 hours · Prerequisites: 02.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You show marketing the line from 02.2: *spend = 12.5 × visits + 25*. The head of marketing asks three questions in a row:

1. *"How far off is it, typically?"*
2. *"What is the worst case I should plan for?"*
3. *"Is this actually any good, or would guessing the average do the same?"*

"The sum of squared residuals is 150" answers none of them. It is in squared pounds, it grows with the number of customers, and it has no scale: is 150 good? You need numbers a non-technical person can act on, and you need to know which number to show to whom.

## 2. The idea

All four metrics start from the same list: the residuals $y - \hat{y}$. They differ only in how they summarise it.

| Question | Metric | How it summarises the residuals |
|---|---|---|
| Typically how far off? | **MAE** | average of the absolute values |
| Penalise big misses more? | **MSE** | average of the squares |
| …but in pounds, not pounds² | **RMSE** | square root of MSE |
| Better than guessing the mean? | **R²** | 1 − (our squared error ÷ the baseline's squared error) |

MAE is what you tell marketing. RMSE is what you watch while training (it is what least squares minimises). R² is what you put in the summary slide, because it is the only one that means the same thing on every dataset: 0 is "no better than the mean", 1 is perfect.

## 3. Definition

For $n$ rows with truth $y_i$, prediction $\hat{y}_i$ and mean $\bar{y}$:

- **MAE (mean absolute error)**: average size of the miss, in the target's units. Every pound counts the same.
- **MSE (mean squared error)**: average squared miss. A £10 miss costs 100, a £20 miss costs 400: big errors dominate. Units are pounds².
- **RMSE (root mean squared error)**: $\sqrt{\text{MSE}}$, back in pounds. Always $\ge$ MAE; the gap between them tells you how uneven the errors are.
- **R² (coefficient of determination)**: the fraction of the target's variation the model explains. Compares your squared error with the squared error of the baseline "predict $\bar{y}$". 1 = perfect, 0 = same as the baseline, negative = worse than the baseline (possible, and it happens).

## 4. Formula

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert$$

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \qquad\qquad \text{RMSE} = \sqrt{\text{MSE}}$$

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- $SS_{\text{res}}$: sum of squared residuals, the model's leftover error. The 150 from 02.2.
- $SS_{\text{tot}}$: sum of squared distances from the mean, the error the *baseline* makes. How much variation there was to explain in the first place.
- $R^2$ reads: "of all the variation in spend, the share the line accounts for".

## 5. By hand

The 02.2 line on the five customers. Residuals: 0, 5, −10, 5, 0.

| $y$ | $\hat{y}$ | $y-\hat{y}$ | $\lvert y-\hat{y}\rvert$ | $(y-\hat{y})^2$ | $y-\bar{y}$ | $(y-\bar{y})^2$ |
|---|---|---|---|---|---|---|
| 50 | 50 | 0 | 0 | 0 | −50 | 2500 |
| 80 | 75 | 5 | 5 | 25 | −20 | 400 |
| 90 | 100 | −10 | 10 | 100 | −10 | 100 |
| 130 | 125 | 5 | 5 | 25 | 30 | 900 |
| 150 | 150 | 0 | 0 | 0 | 50 | 2500 |
| | | **sum** | **20** | **150** | | **6400** |

- $\text{MAE} = 20/5 = \mathbf{4}$. "Typically £4 off."
- $\text{MSE} = 150/5 = \mathbf{30}$ pounds².
- $\text{RMSE} = \sqrt{30} = \mathbf{5.48}$. Bigger than MAE because one £10 miss weighs more when squared.
- $R^2 = 1 - 150/6400 = 1 - 0.0234 = \mathbf{0.977}$. The line explains 97.7% of the variation in spend.

And the baseline, predicting $\bar{y} = 100$ for everyone: residuals −50, −20, −10, 30, 50.

- MAE = 160/5 = **32**. MSE = 6400/5 = **1280**. RMSE = **35.8**. $R^2 = 1 - 6400/6400 = \mathbf{0}$, by construction.

So marketing's three answers: typically £4 off; worst miss £10, and RMSE £5.5 says misses are fairly even; and yes, it is good: the baseline is £32 off and the line explains 98% of the variation.

## 6. In code

[lesson.ipynb](lesson.ipynb) computes the table in NumPy, then with scikit-learn's functions, and the numbers match:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae  = mean_absolute_error(y, y_hat)        # 4.0
mse  = mean_squared_error(y, y_hat)         # 30.0
rmse = np.sqrt(mse)                         # 5.477
r2   = r2_score(y, y_hat)                   # 0.9766
```

Then it shows **why the metrics disagree**: it adds one wholesaler to the table and watches MAE move a little while RMSE jumps, and R² for the baseline go negative when the "model" is a bad guess.

## 7. On real data

On the 3,317-customer table (target = next-quarter spend), the notebook scores three predictors: the mean baseline, the median baseline, and the one-feature line *spend_next ≈ 0.54 × past_spend − 71*. The line cuts MAE and RMSE by about 40% and reaches an R² of about 0.67, while the median baseline, although it has a *lower MAE* than the mean baseline, has a *negative R²*. That is not a contradiction; it is the two metrics measuring different things, and the notebook explains which to believe for which decision.

## 8. Where it breaks

- **MAE hides the worst case.** Two models with MAE £100: one misses every customer by £100, the other is perfect on 99% and off by £10,000 on 1%. Same MAE. RMSE separates them.
- **RMSE is hostage to outliers.** One wholesaler can double it. If your business cares about typical customers, report MAE (or the median absolute error) alongside.
- **R² is not "percent correct".** It is a comparison with the baseline. A high R² on a target with huge natural variation can still mean large pound errors. Report R² *with* MAE, never alone.
- **R² on training data flatters.** It can only go up as you add features, even useless ones. Module 04 fixes this with held-out data.
- **All four are averages over rows.** They say nothing about *which* rows are wrong. Always plot residuals against the prediction.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: a model predicts 60, 70, 100, 120, 160 for the five customers. Compute MAE, MSE, RMSE, R². Better or worse than the line?
2. In code: for the five customers, find the constant prediction $k$ that minimises MAE, and the constant that minimises MSE. (One is the median, the other the mean. Show it.)
3. On the 20-customer table, score the 02.2 line (visits → spend) and a line using `items_last_month`. Which wins on MAE? On RMSE? Are they the same winner?
4. Add one customer with 3 visits and £5,000 spend to the 20-row table and re-score the visits line. Report the % change in MAE, RMSE and R².

**Quiz**

1. Which metric is in the same units as the target: MSE, RMSE, or R²? (Two answers.)
2. RMSE is always ≥ MAE. When are they equal?
3. What does $R^2 = 0$ mean? And $R^2 < 0$?
4. Marketing asks "how far off, typically?" Which metric and why?
5. Which metric does least squares (02.2) minimise?

## 10. Recap

- All metrics summarise the residuals $y - \hat{y}$.
- **MAE**: typical miss in pounds. **RMSE**: typical miss that punishes big errors, in pounds. **MSE**: RMSE², used for training. **R²**: share of variation explained, 0 = baseline, 1 = perfect.
- On the five customers: MAE 4, MSE 30, RMSE 5.48, R² 0.977; baseline MAE 32, R² 0.
- Report **MAE and R² together**; watch RMSE for outlier damage; always plot residuals.

**Next:** [02.4 Gradient descent](../04-gradient-descent/) finds the line that minimises MSE without the formula, the way neural networks do.
