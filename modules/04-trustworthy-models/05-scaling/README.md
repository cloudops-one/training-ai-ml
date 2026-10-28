# 04.5 · Scaling features: who needs it and who does not

⏱ about 1 hour · Prerequisites: 02.4 (gradient descent), 03.2 (KNN), 04.3 (log transform) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention asks a simple question about the KNN churn model: *"Which features does it actually use?"* You have given it five: orders, spend, products, tenure and days since last order. You check, and the honest answer is: **one**. Spend.

Nobody told the model to ignore the others. It happened because `spend` is measured in pounds (typical values in the hundreds and thousands) while `orders` is measured in single digits. A KNN model measures similarity by distance, and in a distance, big numbers win. The units you happened to store the data in decided what the model listens to.

This lesson shows how to put features on a common scale, which models need it, which do not, and the one rule that keeps scaling from leaking.

## 2. The idea

Imagine choosing a flatmate by comparing candidates on two things: monthly income in pounds and number of pets. Two candidates earning £2,000 and £2,020 with 0 and 6 pets differ by "20" on income and "6" on pets, so by raw numbers income is the bigger difference. That is absurd: £20 is nothing; six pets is a lifestyle.

The fix is to measure each difference **relative to how much that feature usually varies**. Incomes typically vary by hundreds of pounds, so £20 is tiny; pet counts typically vary by one or two, so 6 is huge. Dividing by the typical spread puts both on the same footing.

## 3. Definition

- **Feature scaling**: transforming numeric features so they have comparable ranges.
- **Standardisation** (z-scoring, `StandardScaler`): subtract the mean, divide by the standard deviation. Result: mean 0, standard deviation 1.
- **Min-max scaling** (normalisation, `MinMaxScaler`): subtract the minimum, divide by the range. Result: values between 0 and 1.
- **Scale-sensitive model**: one whose behaviour depends on the units of the features, because it uses distances, dot products with penalties, or gradient descent.
- **Scale-invariant model**: one whose predictions do not change if a feature is multiplied by a constant, such as a decision tree.

## 4. Formula

$$\text{standardise:}\quad z = \frac{x - \mu}{\sigma} \qquad\qquad \text{min-max:}\quad x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

- $\mu$, $\sigma$: mean and standard deviation of the feature **in the training data**. `StandardScaler` uses $\sigma = \sqrt{\frac{1}{n}\sum (x-\mu)^2}$ (dividing by $n$, not $n-1$).
- $x_{\min}$, $x_{\max}$: minimum and maximum **in the training data**.

Euclidean distance between customers $a$ and $b$ over features $j$:

$$d(a, b) = \sqrt{\sum_j (a_j - b_j)^2}$$

Each feature's contribution is its squared difference. If one feature's differences are a thousand times bigger, its squares are a million times bigger.

## 5. By hand

**Five spends:** £200, 400, 600, 800, 1,000.

- Mean $\mu = 3000/5 = 600$. Squared deviations: 160,000, 40,000, 0, 40,000, 160,000; sum 400,000; divide by 5 → 80,000; $\sigma = \sqrt{80{,}000} = 282.84$.
- z-scores: $(200-600)/282.84 = -1.414$, then $-0.707$, $0$, $0.707$, $1.414$.
- Min-max: $(x-200)/800$ = 0, 0.25, 0.5, 0.75, 1.

(Pandas' `.std()` divides by $n-1$ and gives 316.23 instead. Small difference, real trap.)

**Who is A's nearest neighbour?**

| customer | orders | spend (£) |
|---|---|---|
| A | 2 | 500 |
| B | 12 | 520 |
| C | 2 | 900 |

Unscaled:

- $d(A, B) = \sqrt{(12-2)^2 + (520-500)^2} = \sqrt{100 + 400} = 22.36$
- $d(A, C) = \sqrt{0 + 400^2} = 400$

B looks like A's twin. But B orders six times as often; C behaves like A and just spent more.

Scaled with the real table's standard deviations (orders 5.6, spend £5,974):

- $d(A, B) = \sqrt{(10/5.6)^2 + (20/5974)^2} = \sqrt{3.19 + 0.00001} \approx 1.79$
- $d(A, C) = \sqrt{0 + (400/5974)^2} = 0.067$

Now C is the nearest neighbour. Ten extra orders is almost two typical spreads; £400 is a fraction of one.

## 6. In code

Open [lesson.ipynb](lesson.ipynb).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaler.fit(X_train)                       # learns mean_ and scale_ from TRAIN rows
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)       # test rows use train's mean and std
new_s = scaler.transform(new_customer)    # so does every live prediction

print(scaler.mean_, scaler.scale_)        # what it learned
```

## 7. On real data

- **Unscaled KNN listens to one feature.** Summed over 2,000 random pairs of training customers, **99.93%** of the squared distance comes from `spend`; orders, products, tenure and recency share the remaining 0.07%.
- **Accuracy (KNN, test set):**

| k | unscaled, 5 features | spend only | scaled, 5 features |
|---|---|---|---|
| 5 | 62.0% | 62.5% | 61.4% |
| 15 | 66.0% | 64.2% | 64.5% |
| 51 | 67.7% | 65.2% | 67.8% |

The unscaled model is close to the spend-only model, as the distance shares predict. Scaling does *not* raise accuracy here; spend happens to be a good feature, so letting it dominate was a lucky accident. Scaling removes the accident; it does not guarantee a gain.
- **Other models:** SVM (RBF kernel) 66.4% unscaled → **68.3%** scaled. Logistic regression 67.2% → 67.5%, but its solver needs **99** iterations unscaled versus **12** scaled. Decision tree: 65.1% vs 64.9%, essentially unchanged.
- **A new customer** (3 orders, £900, …) is scaled with the training mean (3.4 orders) and standard deviation (5.2 orders): $(3 - 3.4)/5.2 \approx -0.07$.

## 8. Where it breaks

**Who needs scaling?**

| needs it | why | doesn't need it | why |
|---|---|---|---|
| KNN, K-Means | distances | decision trees | splits on thresholds; order is unchanged by scaling |
| SVM | distances / margins | random forests | made of trees |
| logistic / linear regression with regularisation (Ridge, Lasso, 04.10) | the penalty treats all coefficients equally | gradient-boosted trees (XGBoost, LightGBM) | made of trees |
| anything trained by gradient descent, incl. neural networks | badly scaled features make the loss a narrow valley (02.4) | plain linear regression solved exactly | coefficients rescale; predictions unchanged |
| PCA | finds directions of largest variance | Naive Bayes (Gaussian) | models each feature separately |

Common mistakes:

- **Fitting the scaler on all data, or on the test data.** The mean and standard deviation are learned parameters; learn them on train only (04.2).
- **Min-max with outliers.** One wholesaler sets the maximum and squeezes everyone else towards 0. Log-transform first (04.3), or standardise, or use `RobustScaler` (median and IQR).
- **Scaling one-hot columns or IDs.** 0/1 columns are already on a sensible scale; customer IDs should not be features at all.
- **Forgetting to scale at prediction time.** The live system must apply the *same fitted* scaler. A `Pipeline` (04.8) bundles scaler and model so this cannot be forgotten.
- **Expecting scaling to improve accuracy.** It makes features comparable. Whether that helps depends on the features.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: standardise and min-max scale orders = 1, 2, 3, 4, 10. Check with scikit-learn.
2. Min-max scale the real `spend` column, with and without `log1p`. How many customers end up near 0?
3. Fit separate scalers on train and test (the mistake) and compare with the right way. Why is it worse still for a single live customer?
4. Log-transform orders, spend and products before scaling and re-run KNN.

**Quiz**

1. Write the formulas for standardisation and min-max scaling. What range or distribution does each produce?
2. Why does an unscaled feature measured in pounds dominate a KNN model?
3. Name three kinds of model that need scaling and two that do not.
4. On which rows is a scaler fitted, and which rows is it applied to?
5. Spend has one customer at £176,000 and a median of £553. What goes wrong with min-max scaling?

Answers are at the end of the lesson notebook.

## 10. Recap

- Distances, penalties and gradient descent all see raw numbers, so features in big units dominate.
- **Standardise** ($z = (x-\mu)/\sigma$) or **min-max** ($[0, 1]$); log-transform skewed money columns first.
- KNN, SVM, regularised linear models, neural networks, PCA and K-Means need scaling. Trees and tree ensembles do not.
- Fit the scaler on training rows; apply the same fitted scaler to test and live data.
- Scaling decides which features a model can hear. It does not promise higher accuracy.

**Next:** [04.6 Feature engineering](../06-feature-engineering/): creating better features from dates, ratios and customer behaviour.
