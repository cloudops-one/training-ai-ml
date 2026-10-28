# 02.2 · Linear regression by hand: the best straight line

⏱ about 2 hours · Prerequisites: 02.1, mean (01.8) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing wants to know how much each customer will spend next month. You have one fact that looks useful: how many times the customer visited the website last month. Here are five customers whose next-month spend you already know, because the month has passed:

| customer | visits last month ($x$) | spend next month, £ ($y$) |
|---|---|---|
| C001 | 2 | 50 |
| C002 | 4 | 80 |
| C003 | 6 | 90 |
| C004 | 8 | 130 |
| C005 | 10 | 150 |

A new customer, C006, visited **7** times. What will they spend?

You could guess "somewhere between 90 and 130". A model should do better: it should give one number, and give it the same way for 40,000 customers.

## 2. The idea

Plot the five points. They are not on a perfect line, but they are close to one. More visits, more spend, roughly in proportion.

```
spend
 150 |                                   •
 130 |                           •
 100 |
  90 |                   •
  80 |           •
  50 |   •
     +---+-------+-------+-------+-------+----
         2       4       6       8      10   visits
```

If you drew a ruler through the middle of those points, you could read off a prediction for 7 visits. **Linear regression is the recipe for placing the ruler in the best possible position.** "Best" has a precise meaning: the line whose total squared miss across all points is as small as possible.

Why a line? Because it is the simplest shape that captures "more of this means more of that", it has only two numbers to learn, and you can explain it to anyone: *"every extra visit is worth about £12.50"*.

## 3. Definition

**Linear regression** fits a straight line

$$\hat{y} = m\,x + c$$

to data, choosing the **slope** $m$ and **intercept** $c$ that minimise the sum of squared errors between predictions $\hat{y}$ and true values $y$.

- **Slope $m$**: how much $y$ changes when $x$ goes up by 1. For ShopWave: pounds of extra spend per extra visit.
- **Intercept $c$**: the predicted $y$ when $x = 0$. Spend of a customer who never visited (often not meaningful on its own, it just positions the line).
- **Residual**: $y - \hat{y}$ for one row. How far the point sits above (+) or below (−) the line.
- **Least squares**: the rule "pick the line that makes the sum of squared residuals smallest". Squared, so that misses above and below do not cancel, and so that big misses hurt more than small ones.

## 4. Formula

With $\bar{x}$ the mean of the $x$ values and $\bar{y}$ the mean of the $y$ values, the least-squares line is:

$$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\qquad\qquad
c = \bar{y} - m\,\bar{x}$$

In words:

- **Numerator of $m$**: for each row, how far $x$ is from its mean, times how far $y$ is from its mean, summed. When $x$ and $y$ are both above average (or both below) the product is positive. This measures *how they move together*.
- **Denominator of $m$**: how spread out $x$ is, squared and summed. It rescales the numerator into "units of $y$ per unit of $x$".
- **$c$**: forces the line to pass through the point $(\bar{x}, \bar{y})$, the centre of the data. The best line always does.

These formulas are not pulled from the air. They are what you get if you ask calculus "where is the sum of squared errors smallest?". Module 02.4 shows an alternative route (gradient descent) that finds the same answer by trial and improvement, and that route is what neural networks use.

## 5. By hand

**Step 1: means.**

$\bar{x} = (2+4+6+8+10)/5 = 30/5 = 6$

$\bar{y} = (50+80+90+130+150)/5 = 500/5 = 100$

**Step 2: deviations and their products.**

| $x$ | $y$ | $x-\bar{x}$ | $y-\bar{y}$ | $(x-\bar{x})(y-\bar{y})$ | $(x-\bar{x})^2$ |
|---|---|---|---|---|---|
| 2 | 50 | −4 | −50 | 200 | 16 |
| 4 | 80 | −2 | −20 | 40 | 4 |
| 6 | 90 | 0 | −10 | 0 | 0 |
| 8 | 130 | 2 | 30 | 60 | 4 |
| 10 | 150 | 4 | 50 | 200 | 16 |
| | | | **sum** | **500** | **40** |

**Step 3: slope.**

$m = 500 / 40 = 12.5$

Every extra visit is worth £12.50 of next-month spend.

**Step 4: intercept.**

$c = \bar{y} - m\bar{x} = 100 - 12.5 \times 6 = 100 - 75 = 25$

**Step 5: the model.**

$$\hat{y} = 12.5\,x + 25$$

**Step 6: predict for C006 (7 visits).**

$\hat{y} = 12.5 \times 7 + 25 = 87.5 + 25 = 112.5$

So the model says C006 will spend about **£112.50**.

**Step 7: how good is the line?** Compute each prediction and residual.

| $x$ | $y$ | $\hat{y} = 12.5x + 25$ | residual $y - \hat{y}$ | residual² |
|---|---|---|---|---|
| 2 | 50 | 50 | 0 | 0 |
| 4 | 80 | 75 | 5 | 25 |
| 6 | 90 | 100 | −10 | 100 |
| 8 | 130 | 125 | 5 | 25 |
| 10 | 150 | 150 | 0 | 0 |
| | | | **sum** | **150** |

Sum of squared residuals = 150. **No other straight line gives a smaller number.** That is what "best" means. (Try $m = 12, c = 28$ in the notebook and watch the sum go up.)

Note the residuals sum to zero (0 + 5 − 10 + 5 + 0). They always do for a least-squares line: the misses above and below balance exactly.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). It does the same steps three times, and the numbers match each time:

**Plain Python** (the by-hand table, typed in):

```python
x = [2, 4, 6, 8, 10]
y = [50, 80, 90, 130, 150]
x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)
m = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) \
  / sum((xi - x_mean) ** 2 for xi in x)
c = y_mean - m * x_mean
print(m, c)            # 12.5 25.0
```

**NumPy** (same formula, no loop):

```python
import numpy as np
x, y = np.array(x), np.array(y)
m = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
c = y.mean() - m * x.mean()
```

**scikit-learn** (what you will use from now on):

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)      # X must be 2-D: rows × features
print(model.coef_[0], model.intercept_)   # 12.5 25.0
model.predict([[7]])                       # array([112.5])
```

`coef_` is $m$, `intercept_` is $c$, `fit` runs the formula, `predict` computes $m x + c$. Nothing is hidden; you have just done all of it by hand.

## 7. On real data

The notebook then loads `shopwave_customers_small.csv` (20 customers) and fits visits → spend. The slope will not be a tidy 12.5 any more, and the points will scatter more. You will plot the line over the points and read the residuals. Then it fits the same thing on the real Online Retail data (thousands of customers, visits replaced by number of orders) so you see the method does not care about size.

## 8. Where it breaks

- **The relationship is not a line.** Spend might rise with visits and then flatten (nobody spends £10,000 after 800 visits). A line will be wrong at the ends. Lesson 02.6 adds curves.
- **One wild point drags the line.** Because errors are *squared*, a customer who visited 3 times and spent £5,000 pulls the slope hard. Always plot before you trust a slope.
- **Extrapolation.** The line was fitted on 2–10 visits. Predicting for 50 visits is a guess the data never supported.
- **Correlation is not cause.** The slope says visits and spend move together, not that forcing people to visit more would make them spend more.
- **One feature is rarely enough.** Lesson 02.5 uses many at once; the formula generalises cleanly.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: fit a line to $x = [1, 2, 3, 4]$, $y = [3, 5, 7, 9]$. (It fits perfectly. What are $m$, $c$ and the sum of squared residuals?)
2. By hand, then check in code: change C003's spend from 90 to 190. What happens to $m$ and $c$? Which point is now the biggest residual?
3. In code: fit `items_last_month` → `spend_next_month` on the 20-customer table. Is the slope bigger or smaller than for visits? What does that mean in pounds?
4. Try to beat least squares: pick any $m$ and $c$ you like and compute the sum of squared residuals. Can you get below 150 on the five-row table?

**Quiz**

1. In $\hat{y} = m x + c$, what does $m$ mean for ShopWave in plain English?
2. Why square the residuals instead of just adding them?
3. The best line always passes through one particular point. Which?
4. `model.coef_` and `model.intercept_`: which is $m$ and which is $c$?
5. The line predicts £112.50 for 7 visits. Give one reason the real number might be very different.

## 10. Recap

- Linear regression predicts with a line $\hat{y} = m x + c$; $m$ is the change in $y$ per unit of $x$, $c$ positions the line.
- "Best" line = smallest **sum of squared residuals**. That is the least-squares rule.
- $m = \dfrac{\sum (x-\bar{x})(y-\bar{y})}{\sum (x-\bar{x})^2}$, then $c = \bar{y} - m\bar{x}$. Five rows, one small table, done by hand.
- scikit-learn's `LinearRegression().fit(X, y)` runs exactly this; `coef_` and `intercept_` are $m$ and $c$.
- Always plot. Lines fail on curves, outliers and extrapolation.

**Next:** [02.3 Measuring error](../03-measuring-error/) turns "sum of squared residuals = 150" into numbers you can report: MAE, RMSE and R².
