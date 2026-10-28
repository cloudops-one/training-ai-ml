# 02.5 · Many features: multiple linear regression

⏱ about 1.5 hours · Prerequisites: 02.2, 02.3 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The visits line from 02.2 explains spend fairly well on the 20 customers, but look at who it gets wrong: C001 visited twice and spent far less than predicted; C015 visited eleven times and spent less than the line expects. Marketing points out the obvious: *"You know how many items they bought and how long since their last order. Why are you ignoring that?"*

One feature, one slope. Several features, several slopes. The question is how to find them all at once, and, harder, how to **read** them once you have them.

## 2. The idea

A line through a cloud of points becomes a flat sheet through a cloud of points in 3-D (two features) and a "hyperplane" in more dimensions, which nobody can picture but which behaves exactly the same way: each feature gets its own slope, and the prediction is the sum of the contributions plus an intercept.

```
 spend = 38  +  12.6 × visits  +  7.7 × items  −  0.7 × days_since_last
           ▲        ▲                ▲                 ▲
      start here   per visit        per item     per day of silence
```

Reading a slope: *"holding every other feature fixed, one more unit of this feature changes the prediction by this much."* That phrase, *holding the others fixed*, is what makes multiple regression different from running three separate one-feature lines, and it is the source of every misreading.

## 3. Definition

**Multiple linear regression** predicts a number as a weighted sum of $p$ features:

$$\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_p x_p$$

- $b_0$: the **intercept**, the prediction when every feature is 0.
- $b_j$: the **coefficient** (slope) of feature $j$: the change in $\hat{y}$ for a one-unit increase in $x_j$, with all other features held constant.
- The coefficients are chosen by **least squares**, the same rule as 02.2: minimise $\sum (y_i - \hat{y}_i)^2$.

The closed-form answer exists (the "normal equation", a matrix formula) and scikit-learn uses it; gradient descent from 02.4 reaches the same answer with one gradient per coefficient. You do not need the matrix form in this course.

A **categorical** feature (country) has no "one unit". It is converted into 0/1 columns, one per category minus one, called **one-hot encoding** (full treatment in 04.4). Each such coefficient reads: "the difference from the left-out category, holding everything else fixed".

## 4. Formula

For a customer with visits $v$, items $k$ and days since last order $d$:

$$\hat{y} = b_0 + b_v\, v + b_k\, k + b_d\, d$$

Least squares picks $(b_0, b_v, b_k, b_d)$ to minimise $\sum_i (y_i - \hat{y}_i)^2$ over all 20 customers. Everything from 02.3 (MAE, RMSE, R²) applies unchanged.

## 5. By hand

scikit-learn fits the 20-customer table (you will run it in the notebook) and reports:

$$\hat{y} = 38.1 + 12.58\,v + 7.65\,k - 0.70\,d$$

**Predict for three customers** by plugging in:

| customer | $v$ | $k$ | $d$ | $12.58v$ | $7.65k$ | $-0.70d$ | $\hat{y}$ (+38.1) | actual $y$ |
|---|---|---|---|---|---|---|---|---|
| C001 | 2 | 2 | 55 | 25.2 | 15.3 | −38.5 | **40.1** | 27.6 |
| C006 | 13 | 7 | 20 | 163.5 | 53.6 | −14.0 | **241.2** | 250.0 |
| C009 | 3 | 3 | 5 | 37.7 | 23.0 | −3.5 | **95.3** | 94.8 |

Compare C001: the visits-only line (17.17 × 2 + 19.6) predicted £54. Knowing C001 has been silent for 55 days pulls the prediction down to £40, closer to the £27.6 they actually spent.

**Reading the coefficients.** With visits alone, the slope was **17.17** per visit. With items and days added it is **12.58**. Visits did not become less important; part of the credit it was taking now goes to items, because people who visit more also buy more items and the model can now tell those effects apart. *A coefficient is only meaningful next to the list of the other features in the model.*

**A categorical feature.** Adding `country` as one-hot columns (DE left out as the reference):

$$\hat{y} = 37.0 + 12.41\,v + 7.63\,k - 0.81\,d + 15.1\,[\text{FR}] + 7.3\,[\text{UK}]$$

Reads: a French customer with the same visits, items and recency as a German one is predicted to spend £15 more; a UK one £7 more. On 20 customers those are weak evidence (and the notebook shows how little they improve the fit); the mechanism is what matters now.

## 6. In code

Nothing new in the API. `X` just has more columns:

```python
X = cust20[["visits_last_month", "items_last_month", "days_since_last"]]
model = LinearRegression().fit(X, y)
pd.Series(model.coef_, index=X.columns)       # one slope per column
model.intercept_
```

For `country`, `pd.get_dummies(..., drop_first=True)` makes the 0/1 columns. The notebook then compares the three-feature model with the visits line on MAE and R², and plots predicted vs actual.

## 7. On real data

On the 3,317 real customers, the notebook fits next-quarter spend on six features (orders, spend, products, avg_order, tenure_days, days_since_last). R² climbs from about 0.67 (spend alone, 02.3) to about 0.72 and RMSE improves; MAE actually gets slightly *worse*. Six features bought five points of R², mostly on the biggest customers. Then it reads the coefficients and finds one that looks *wrong*: `days_since_last` has a **positive** slope, as if customers who have been silent longer spend *more*. They do not. The coefficient is positive because, holding `spend` and `orders` fixed, the remaining customers with long gaps are the big infrequent wholesalers. "Holding the others fixed" is doing exactly what it says, and the lesson is that coefficients describe the model, not the world.

## 8. Where it breaks

- **Correlated features share credit unpredictably.** `orders`, `products` and `items` all measure "how much they buy". Put all three in and the coefficients can flip sign while the predictions stay fine. Called **multicollinearity**; 01.9's correlation heat-map is the warning sign; Lasso (04.10) is one cure.
- **Coefficients are in the feature's units.** A slope of 0.49 on `spend` (£) and 15.5 on `orders` (count) says nothing about which matters more. Scale first (04.5) if you want to compare.
- **Adding features never lowers training R².** Even random noise columns raise it a little. That is why the project uses a held-out set (02.6 introduces it, 04.1 makes it law).
- **Linearity is still assumed** per feature. If spend rises with visits and then flattens, a slope cannot say so. 02.6 adds curves.
- **"Holding others fixed" may describe nobody.** A customer with 14 visits and 0 items does not exist; the coefficient for visits is still computed as if they could.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: using the three-feature equation, predict for a new customer with 9 visits, 4 items and 30 days since last order. Then for the same customer with 60 days. What is the difference, and which coefficient explains it?
2. In code: fit all seven subsets of the three features (each alone, each pair, all three). Tabulate R² and MAE. Which single feature is best? Does adding `items` to `visits` help?
3. Add a column of pure random noise to the 20-row table and refit. Does R² go up or down? By how much? What does that tell you about R² on training data?
4. On the real data, fit with and without `items` (which correlates ~0.9 with `spend`). Compare the coefficient of `spend` in each. Explain the change.

**Quiz**

1. In $\hat{y} = b_0 + b_1 x_1 + b_2 x_2$, what does $b_1$ mean in words?
2. The visits coefficient fell from 17.2 to 12.6 when items was added. Did visits become less important?
3. Why does one-hot encoding leave one category out?
4. A coefficient has an implausible sign. Name two possible reasons.
5. Can adding a useless feature lower the R² computed on the training data?

## 10. Recap

- Several features: $\hat{y} = b_0 + \sum_j b_j x_j$, fitted by least squares; same metrics, same API, more columns.
- Each $b_j$ = change in $\hat{y}$ per unit of $x_j$ **holding the other features fixed**; it depends on which other features are present.
- Categories become 0/1 columns; their coefficients are differences from a reference category.
- On the 20 customers, three features cut MAE from £10.7 to £7.8 and raised R² from 0.96 to 0.98; on the real data, six features added five points of R² but no MAE.
- Correlated features make coefficients unreliable even when predictions are fine; training R² never falls when you add a column.

**Next:** [02.6 Curves and overfitting](../06-polynomial-and-overfitting/) lets the model bend, and shows the first thing that goes badly wrong when it can.
