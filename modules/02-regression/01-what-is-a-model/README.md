# 02.1 · What is a model? Features, target, prediction, error

⏱ about 1 hour · Prerequisites: Module 01 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing has 2,000 printed catalogues and 40,000 customers. *"Send them to the people who will spend the most next month."*

You could rank customers by what they spent *last* month. That is a model, a crude one: *next month = last month*. Marketing tries it, and the catalogues go to a wholesaler who orders once a year and 1,999 people who happened to buy Christmas gifts. It is not a bad start, it is just not very good, and nobody can say *how* not-good it is.

Before building anything better you need four words with precise meanings: **feature**, **target**, **prediction**, **error**, and a way to measure whether any model, including "next month = last month", is worth using.

## 2. The idea

A model is a **machine that turns facts about a thing into a guess about that thing**.

```
   facts about one customer                           guess
 ┌───────────────────────────┐    ┌─────────┐    ┌──────────────┐
 │ visits = 7                │    │         │    │              │
 │ items  = 5                │ ─▶ │  model  │ ─▶ │ spend ≈ £139 │
 │ days since last = 25      │    │         │    │              │
 └───────────────────────────┘    └─────────┘    └──────────────┘
```

The machine has knobs inside. **Training** means turning the knobs until the guesses for customers whose answers we *already know* are as good as possible. Then we trust it on customers whose answers we do not know yet.

That is the whole of supervised learning. Every model in this course, from a straight line to a language model, is this box with different insides.

## 3. Definition

For each row (one customer):

- **Features** $x_1, x_2, \dots, x_p$: the inputs. Numbers the model may look at. Together, one row of features is written $\mathbf{x}$; the whole table of them is $X$ (rows = customers, columns = features).
- **Target** $y$: the true answer for that row. Known for past customers, unknown for new ones.
- **Prediction** $\hat{y}$ ("y-hat"): the model's guess for that row.
- **Error** (or **residual**) for that row: $y - \hat{y}$. Positive means the model guessed too low.
- **Parameters**: the knobs inside the model. For a straight line, the slope and the intercept. Training sets them.
- **Baseline**: the simplest possible model, usually "predict the average of $y$ for everyone". Any model that cannot beat the baseline is not a model, it is a ritual.

A **regression** model is one whose target is a number (spend in pounds). A **classification** model's target is a category (will churn / will not). This module is regression.

## 4. Formula

$$\hat{y} = f(\mathbf{x};\ \theta)$$

- $f$: the model's shape (a line, a tree, a network).
- $\theta$ (theta): its parameters, the knobs.
- $\mathbf{x}$: one row of features.

Training chooses $\theta$ to make the errors $y - \hat{y}$ small across all training rows. "Small" is defined by a **loss function**; the next lesson defines the common ones. For now we use the simplest:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

the **mean absolute error**: on average, how many pounds the guess is off by.

## 5. By hand

Five customers with a known next-month spend. Two candidate models, no training yet:

- **Model A (baseline)**: predict the mean of $y$ for everybody.
- **Model B**: predict $10 \times$ visits.

| customer | visits ($x$) | spend ($y$) | A: $\hat{y}$ = mean | A: $\lvert y-\hat{y}\rvert$ | B: $\hat{y} = 10x$ | B: $\lvert y-\hat{y}\rvert$ |
|---|---|---|---|---|---|---|
| C001 | 2 | 50 | 100 | 50 | 20 | 30 |
| C002 | 4 | 80 | 100 | 20 | 40 | 40 |
| C003 | 6 | 90 | 100 | 10 | 60 | 30 |
| C004 | 8 | 130 | 100 | 30 | 80 | 50 |
| C005 | 10 | 150 | 100 | 50 | 100 | 50 |
| | | | | **sum 160** | | **sum 200** |

$\bar{y} = 500/5 = 100$, so model A predicts 100 for everyone.

- MAE of A = 160 / 5 = **£32**
- MAE of B = 200 / 5 = **£40**

Model B *uses* the feature and is still **worse than the baseline**, because its knob (the 10) is badly set: every guess is too low. Model B has the right shape and the wrong parameters. Training is the process of fixing that, and lesson 02.2 does it: the best line turns out to be $12.5x + 25$ with MAE £4.

Write this down as the first rule of modelling: **always compute the baseline first.** It costs one line, and it is the number every later model must beat.

## 6. In code

[lesson.ipynb](lesson.ipynb) builds this vocabulary in Pandas on the five rows, then on the 20-customer table:

```python
X = cust[["visits_last_month"]]        # features: a DataFrame, rows × columns
y = cust["spend_next_month"]            # target: a Series

baseline = y.mean()                     # model A
y_hat_A = pd.Series(baseline, index=y.index)
y_hat_B = 10 * cust["visits_last_month"]   # model B

mae = lambda y, y_hat: (y - y_hat).abs().mean()
print(mae(y, y_hat_A), mae(y, y_hat_B))
```

Then it hands model B's knob to a loop that tries slopes from 5 to 25 and plots MAE against the slope. The bottom of that curve is a trained parameter. You will have trained your first model without a library.

## 7. On real data

On the real ShopWave customer table from lesson 01.9 (3,317 customers, target = spend in the next quarter), the notebook computes the baseline. The mean next-quarter spend is about £776, but the median is £171 and the maximum £117,634. "Predict the mean" is off by **about £950 per customer on average**, and "predict the median" by about £760. Those are the numbers every model in this module has to beat, and the best line in 02.5 gets the error down to about £550. Not £50. Real data is hard, and knowing the baseline keeps you honest about it.

## 8. Where it breaks

- **Features must be known at prediction time.** "Spend next month" cannot be a feature for predicting spend next month. Obvious here; subtle in practice (lesson 01.9 found such a leak, `recency_days`). Ask of every column: *would I have this number on the day I make the prediction?*
- **The target must be the thing you actually want.** Marketing wants *who to send catalogues to*. Predicted spend is a proxy. If 1% of customers are wholesalers, a model that nails them and ignores everyone else has low error and is useless for a consumer catalogue.
- **One row per thing you predict.** The transaction table has one row per invoice line. The model needs one row per *customer*. Getting the unit of analysis right (01.9) comes before any modelling.
- **A model with the right shape and wrong knobs is worse than no model.** Model B proved it. Never ship untrained defaults.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: add model C, $\hat{y} = 15x$. Compute its MAE on the five rows. Is it better than A? Than B?
2. In code: extend the slope search to also search the intercept (two nested loops, slopes 5–20, intercepts 0–50). What pair gives the lowest MAE? Compare with 02.2's least-squares answer $(12.5, 25)$. Is the winner unique?
3. On the 20-customer table, which single column, used as $\hat{y} = k \cdot \text{column}$ with the best $k$, gives the lowest MAE? Is it the column you expected?
4. List three columns in the raw transaction table that must **not** be used as features for next-quarter spend, and say why for each.

**Quiz**

1. What is the difference between $y$ and $\hat{y}$?
2. Give the baseline model for regression in four words.
3. Model B uses the feature and still loses to the baseline. What is wrong with it?
4. Why is `spend_next_quarter` not allowed as a feature?
5. In $\hat{y} = f(\mathbf{x}; \theta)$, which symbol does training change?

## 10. Recap

- A model is a function $f(\mathbf{x};\theta)$: features in, prediction out, parameters $\theta$ set by training.
- **Target** $y$ is the truth, **prediction** $\hat{y}$ the guess, **error** $y - \hat{y}$ the gap.
- **MAE** = mean of $|y - \hat{y}|$: average pounds off.
- The **baseline** (predict the mean) is the first number you compute and the one every model must beat.
- A model with good shape and bad parameters can lose to the baseline. Training exists to fix the parameters.

**Next:** [02.2 Linear regression by hand](../02-linear-regression-by-hand/) finds the best slope and intercept with a formula instead of a loop.
