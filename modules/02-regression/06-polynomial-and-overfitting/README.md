# 02.6 · Curves: polynomial regression and the first look at overfitting

⏱ about 1.5 hours · Prerequisites: 02.3, 02.5 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing ran twelve ad campaigns last year, from £500 to £10,000, and counted the extra orders each one brought. They want to know what a £12,000 campaign would do.

You fit a line. It says every £1,000 brings about 15 more orders, forever. Marketing is not convinced: *"The first thousand pounds brought over 60 orders. The last thousand brought 9. It flattens."* They are right. The relationship is a **curve**, and a line cannot bend.

So you need a model that can bend. And the moment a model can bend, it can bend *too much*, and that failure, **overfitting**, is the central problem of machine learning. You meet both in this lesson.

## 2. The idea

A line is $\hat{y} = b_0 + b_1 x$. Add a squared term and it can bend once: $\hat{y} = b_0 + b_1 x + b_2 x^2$. Add a cubed term and it can bend twice. A degree-9 polynomial can bend eight times, enough to pass through every one of twelve points.

The trick: you do not need a new algorithm. **Make new columns** $x^2, x^3, \dots$ and run the same multiple regression from 02.5. The model is still linear *in its coefficients*; it is only curved in $x$.

```
 degree 1:  x                       →  a line        (cannot bend)
 degree 2:  x, x²                   →  one bend      (diminishing returns)
 degree 9:  x, x², x³, …, x⁹        →  eight bends   (can hit every point)
```

The question is never "can the model fit the data?"; a high enough degree always can. The question is "does it fit **new** data?", and the only way to answer it is to hold some data back and check.

## 3. Definition

- **Polynomial regression**: linear regression on the features $x, x^2, \dots, x^d$. $d$ is the **degree**; it is a hyperparameter (chosen by you, not learned).
- **Underfitting**: the model is too simple to capture the pattern. Training error *and* test error are both high. The line on the curved campaign data.
- **Overfitting**: the model is flexible enough to fit noise in the training data. Training error is very low; test error is high. The degree-9 polynomial.
- **Training set**: rows the model is fitted on. **Test set**: rows held back, never used in fitting, scored once at the end. The honest measure of a model is its **test error**.
- **Generalisation**: doing well on data you did not train on. The entire point.

## 4. Formula

$$\hat{y} = b_0 + b_1 x + b_2 x^2 + \dots + b_d x^d$$

For $d = 2$ the coefficient $b_2$ has a direct reading: negative means the curve bends *down* (each extra unit of $x$ adds less than the last, diminishing returns); positive means it bends up (accelerating returns).

The fitting rule is still least squares, now over $d+1$ coefficients. Nothing else changes.

## 5. By hand

**Features.** For $x = 3$, the degree-3 feature row is $[1,\ 3,\ 9,\ 27]$. For $x = 10$: $[1,\ 10,\ 100,\ 1000]$. That is all `PolynomialFeatures` does. (Notice how fast the columns grow; the notebook scales $x$ first so that $x^9$ does not reach a billion and break the arithmetic.)

**The experiment.** Twelve campaigns to train on; thirty campaigns from this year, with the same underlying curve and fresh noise, to test on.

| degree | training MAE | **test MAE** | prediction for £12k |
|---|---|---|---|
| 1 | 5.9 | 8.9 | 233 |
| 2 | 4.0 | **6.7** | 202 |
| 3 | 3.6 | 6.5 | 216 |
| 5 | 3.6 | 6.3 | 318 |
| 9 | **0.8** | 7.9 | **9,632** |

Read down the columns:

- **Training MAE only falls** as degree rises. It must: a bigger model can do everything a smaller one can.
- **Test MAE falls, then rises.** Degree 1 is too stiff (underfit). Degrees 2–5 are about right. Degree 9 has started fitting the noise (overfit): it is worse on new campaigns than a simple quadratic, despite a training error ten times smaller.
- **Beyond the data** the difference is not subtle. The line says 233 extra orders for £12k; the quadratic 202; the degree-9 polynomial says **9,632**, because between and beyond the training points it is doing whatever the noise told it to.

If you had chosen by training error, you would have shipped degree 9.

## 6. In code

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, y_train)
mean_absolute_error(y_test, model.predict(X_test))
```

`make_pipeline` chains steps so that `fit` scales, expands and fits in one call, and `predict` repeats the same transformations on new rows. Lesson 04.8 makes pipelines the standard way to build every model.

The notebook draws degrees 1, 2 and 9 over the points, tabulates train vs test MAE for degrees 1 to 9, and plots the famous U-shaped curve: training error always falling, test error with a bottom.

## 7. On real data

Does bending help on next-quarter spend? The notebook holds out 20% of the 3,317 customers with `train_test_split` and fits degrees 1 to 5 on `spend`. Training R² rises with degree, as it must. Test R² improves slightly up to degree 3 and then collapses (degree 5 is worse than the mean baseline) because the polynomial goes wild on the wholesalers at the far right. On this target a straight line in past spend is nearly as good as any curve; the test set decides.

## 8. Where it breaks

- **Polynomials explode outside the data.** The degree-9 prediction of 9,632 is the rule, not the exception. Never extrapolate with a high-degree polynomial.
- **Degree is a hyperparameter**, and you just chose it by looking at the test set. Do that repeatedly and the test set stops being honest: you have fitted *to* it. Module 04 adds a validation set and cross-validation so the test set is touched once.
- **Raw powers are numerically ugly.** $10^9$ next to $0.5^9$ gives least squares an ill-conditioned problem. Scale first (the pipeline does).
- **Polynomials in many features multiply.** Degree 2 on 6 features makes 28 columns; degree 3 makes 84. Trees (02.7) and gradient boosting (Module 05) capture curves and interactions without this blow-up.
- **More data shrinks overfitting.** Exercise 3 shows degree 9 behaving on 120 campaigns. Overfitting is a ratio of model freedom to data, not a property of the model alone.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: the degree-3 feature row for $x = 2$ and for $x = 0.5$. Confirm with `PolynomialFeatures(3)`.
2. Fit degree 2 without the scaler and write the equation $\hat{y} = b_0 + b_1 x + b_2 x^2$. What is the sign of $b_2$ and what does it say about diminishing returns?
3. Generate 120 training campaigns from the same curve. Refit degree 9 and compare its test MAE with the 12-row fit. Why did it change?
4. "Test" on the training rows instead. Which degree wins? Why is that answer worthless?

**Quiz**

1. Polynomial regression is still "linear regression". In what sense?
2. Why can training error never rise when you increase the degree?
3. How do you tell underfitting from overfitting using train and test error?
4. What is the one honest way to choose model complexity?
5. A degree-9 polynomial fits all 12 training points almost exactly. Should you trust its prediction for £12k?

## 10. Recap

- Curves via new columns $x^2, x^3, \dots$; same least squares, same API, one hyperparameter: the degree.
- **Underfit**: too simple, both errors high. **Overfit**: too flexible, training error tiny, test error worse.
- Training error always falls with complexity; **test error has a bottom**. Choose the bottom.
- On the campaigns: degree 2 wins on new data (MAE 6.7); degree 9 has training MAE 0.8 and predicts 9,632 orders for £12k.
- From now on, every score you report is on data the model did not see.

**Next:** [02.7 Decision tree regressor](../07-decision-tree-regressor/) bends in a completely different way: steps instead of curves.
