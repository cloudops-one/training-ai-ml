# 03.2 · K-nearest neighbours by hand

⏱ about 1.5 hours · Prerequisites: 03.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A new customer, C021, visited **6 times** last month and last ordered **30 days** ago. Retention asks: voucher or no voucher?

You have 20 customers whose fate you know. The most natural thing a human does is: *"find the customers most like C021 and see what they did."* That instinct, written down precisely, is the k-nearest-neighbours algorithm. It is the simplest classifier there is, it needs no formula to train, and it is still used in production for things like product recommendations and image search.

## 2. The idea

You move to a new town and want to know whether to carry an umbrella. You do not build a weather model; you look at what the five people nearest you are carrying. Three umbrellas, two not: you take one.

KNN does exactly this:

1. Measure the **distance** from the new customer to every known customer.
2. Keep the **k** closest.
3. Let them **vote**. Majority wins.

The whole model is the training data itself. "Training" is just storing the table.

## 3. Definition

**K-nearest neighbours (KNN)** classifies a new point by majority vote among the $k$ training points closest to it in feature space.

- **k**: a hyperparameter you choose. Small k follows local quirks; large k smooths them out.
- **Distance**: how "close" is measured. The default is straight-line (Euclidean) distance.
- **Vote**: with $k = 5$ and three churners among the neighbours, the predicted probability of churn is $3/5 = 0.6$ and the predicted class is churn.

## 4. Formula

**Euclidean distance** between a new point $(a_1, a_2)$ and a known point $(b_1, b_2)$:

$$d = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}$$

It is Pythagoras: the two differences are the legs of a right triangle, the distance is the hypotenuse. With more features you add more squared differences under the root:

$$d = \sqrt{\sum_{j=1}^{p} (a_j - b_j)^2}$$

**Prediction**: among the $k$ smallest distances, count the classes;

$$\hat{p}(\text{churn}) = \frac{\text{churners among the } k}{k}, \qquad \hat{y} = 1 \text{ if } \hat{p} > 0.5$$

## 5. By hand

New customer C021: visits = 6, days since last = 30.

**Step 1: distances to a few customers** (features: visits, days since).

| id | visits | days | $(6 - v)^2$ | $(30 - d)^2$ | sum | $d = \sqrt{\text{sum}}$ | churned |
|---|---|---|---|---|---|---|---|
| C010 | 2 | 28 | 16 | 4 | 20 | **4.47** | 1 |
| C005 | 7 | 25 | 1 | 25 | 26 | **5.10** | 1 |
| C013 | 11 | 28 | 25 | 4 | 29 | **5.39** | 0 |
| C016 | 12 | 29 | 36 | 1 | 37 | **6.08** | 0 |
| C019 | 12 | 34 | 36 | 16 | 52 | **7.21** | 1 |
| C003 | 10 | 22 | 16 | 64 | 80 | 8.94 | 0 |
| C001 | 2 | 55 | 16 | 625 | 641 | 25.32 | 1 |

(The notebook computes all 20; these are the closest, plus one far one to show the scale.)

**Step 2: the 5 nearest** are C010, C005, C013, C016, C019.

**Step 3: vote.** Churned: C010 ✓, C005 ✓, C013 ✗, C016 ✗, C019 ✓ → **3 of 5 churned**. $\hat{p} = 0.6$, prediction: **churn, send the voucher**.

**Step 4: change k.**
- $k = 1$: C010 alone → churn (100%).
- $k = 3$: C010, C005, C013 → 2/3 → churn.
- $k = 7$: add C003 and C008 (both stayed, both at 8.94) → 3/7 → **stay**.

The answer flips between k = 5 and k = 7. C021 sits near the boundary, and k decides how far out we look. There is no "right" k from the formula; you choose it by testing (Module 04 shows how properly).

**Step 5: the scaling trap.** Look at the table again. Days-since-last ranges from 5 to 58; visits from 2 to 14. A difference of 4 visits counts the same as a difference of 4 days, yet 4 visits is a third of the whole range and 4 days is nothing. The distance is dominated by whichever feature has bigger numbers.

Fix: put features on the same scale first. Subtract the mean and divide by the standard deviation (01.8), so every feature has mean 0 and spread 1:

- visits: mean 8.15, std 3.93 → C021: $(6 - 8.15) / 3.93 = -0.55$
- days: mean 30.3, std 15.4 → C021: $(30 - 30.3) / 15.4 = -0.02$

After scaling, the 5 nearest become C005, C020, C017, C010, C003: **2 of 5 churned → stay**. Same data, same k, opposite answer. Scaling is not a detail for KNN; it changes the model. Module 04.5 covers it fully; from now on, scale before any distance-based model.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The distance table in NumPy, in three lines:

```python
new = np.array([6, 30])
dist = np.sqrt(((X20.values - new) ** 2).sum(axis=1))   # one distance per customer
nearest = np.argsort(dist)[:5]                            # indices of the 5 smallest
```

Then scikit-learn:

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X20, y20)                     # "training" = storing the table
knn.predict([[6, 30]])                # → [1]
knn.predict_proba([[6, 30]])          # → [[0.4, 0.6]]
```

And with scaling, as a pipeline so the scaler is never forgotten:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
```

The notebook also draws the decision boundary for k = 1, 5 and 15 so you can *see* k smoothing the map.

## 7. On real data

On the 3,300-customer churn table (five features, 25% held out as test customers), KNN accuracy climbs with k: about 61% at k = 1, 62% at k = 5, 66% at k = 15, 68% at k = 51, against a 56% baseline. The jagged k = 1 boundary memorises noise; larger k averages it away. Scaling changes little here because one feature dominates; you will check that yourself in the exercises.

## 8. Where it breaks

- **Unscaled features.** The silent killer. Always scale.
- **Slow at prediction time.** Every prediction measures distance to every training row. Fine for 3,000 customers, painful for 3 million.
- **Many features.** In 40 dimensions, every point is roughly equally far from every other ("curse of dimensionality", Module 06) and "nearest" stops meaning much.
- **No explanation.** KNN can say "customers like you churned", but not *why*. Trees (03.3) and logistic regression (03.4) can.
- **Ties.** With k = 4 and a 2–2 vote, scikit-learn picks the lower class label. Use odd k for binary problems.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: compute the distance from a customer with visits = 12, days = 10 to C012, C014 and C006. Which is nearest? What would k = 3 predict?
2. In code: on the 20 customers, which k from 1 to 19 (odd values) gives the highest leave-one-out accuracy? (Hint: `cross_val_score` with `cv=LeaveOneOut()`, or loop and drop one row at a time.)
3. On the real table, compare scaled and unscaled KNN at k = 15. Then drop `spend` and compare again. What does that tell you about which feature dominates the distances?
4. KNN can also do regression: predict `spend_next_month` for C021 as the *mean* of its 5 nearest neighbours. Implement it in NumPy, then check with `KNeighborsRegressor`.

**Quiz**

1. What does "training" a KNN model actually do?
2. Write the Euclidean distance between (3, 4) and (0, 0).
3. With k = 5 and neighbours labelled 1, 0, 0, 1, 1, what are $\hat{p}$ and $\hat{y}$?
4. Why does scaling change KNN's predictions but not a decision tree's?
5. Give one reason to prefer a small k and one to prefer a large k.

## 10. Recap

- KNN: measure distance to every known point, take the k nearest, majority vote.
- Distance is Pythagoras extended to any number of features: $\sqrt{\sum (a_j - b_j)^2}$.
- On 20 customers, C021 (6 visits, 30 days) is churn at k = 5 (3/5) and stay at k = 7: k matters, and you choose it by testing.
- **Scale your features** or the one with the biggest numbers decides everything.
- KNN is simple, needs no training, but is slow to predict, cannot explain itself, and suffers in high dimensions.

**Next:** [03.3 Decision tree classifier](../03-decision-tree-classifier/): a model that *can* explain itself, and that you grow by hand with one formula.
