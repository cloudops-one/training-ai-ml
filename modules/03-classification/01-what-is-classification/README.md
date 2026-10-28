# 03.1 · Classification: predicting a label instead of a number

⏱ about 1 hour · Prerequisites: 02.1, 02.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention has budget for 500 win-back vouchers. They want to send them to customers who are *about to stop buying*, not to people who were going to come back anyway. The question for each customer is not "how much?" but "**will they leave: yes or no?**"

You have what you need: for every customer, what they did before a cut-off date, and whether they placed any order in the 90 days after it. That second column is the answer key.

Your first instinct from Module 02 might be: "code *churned* as 1 and *stayed* as 0, and fit a line." Let us see why that does not work, and what does.

## 2. The idea

A regression model answers *how much*. A **classification** model answers *which one*. Same ingredients (features in, a learned function, a prediction out), but the output is a **category**: churn / stay, fraud / fine, mug / plate / bowl.

Picture the 20 ShopWave customers on a chart: visits last month across, days since last order up, churners coloured red, stayers blue. Classification is drawing a **line (or curve, or set of boxes) that separates the colours**. That dividing line is called the **decision boundary**. Every classifier in this module is a different recipe for drawing it.

```
days since
last order
   60 |  R           R
   50 |       R
   40 |          R       B
   30 |  R      R    B   B  B
   20 |      R    B  B    B  B
   10 |  R       B
    0 +----------------------------
      0    3    6    9   12   15   visits last month
               ^
          a boundary at visits ≈ 9 gets most of them right
```

## 3. Definition

**Classification** is supervised learning where the target is a category (a **class**). With two classes it is **binary classification**; the class we care about finding (churn, fraud) is called the **positive class** and coded 1, the other 0.

- **Decision boundary**: the surface in feature space on one side of which the model predicts 1, on the other 0.
- **Majority-class baseline**: always predict the most common class. It is the score any real classifier must beat.
- **Accuracy**: fraction of predictions that are correct. The simplest score; Lesson 03.6 shows why it is often not enough.

## 4. Formula

Only one for now: the baseline.

$$\text{baseline accuracy} = \max(p_1,\; 1 - p_1)$$

where $p_1$ is the fraction of rows in the positive class. If 45% churn, predicting "nobody churns" is right 55% of the time **without looking at a single feature**. A model with 56% accuracy has learned almost nothing.

## 5. By hand

The 20 customers, with the two features we will plot:

| id | visits | days since | churned | | id | visits | days since | churned |
|---|---|---|---|---|---|---|---|---|
| C001 | 2 | 55 | 1 | | C011 | 8 | 47 | 1 |
| C002 | 11 | 44 | 0 | | C012 | 14 | 12 | 0 |
| C003 | 10 | 22 | 0 | | C013 | 11 | 28 | 0 |
| C004 | 7 | 58 | 1 | | C014 | 11 | 8 | 0 |
| C005 | 7 | 25 | 1 | | C015 | 11 | 41 | 0 |
| C006 | 13 | 20 | 0 | | C016 | 12 | 29 | 0 |
| C007 | 2 | 54 | 1 | | C017 | 8 | 20 | 0 |
| C008 | 10 | 22 | 0 | | C018 | 2 | 14 | 1 |
| C009 | 3 | 5 | 1 | | C019 | 12 | 34 | 1 |
| C010 | 2 | 28 | 1 | | C020 | 7 | 40 | 0 |

**Step 1: baseline.** 9 churned, 11 stayed. $p_1 = 9/20 = 0.45$. Predicting "stays" for everyone scores $11/20 = 55\%$.

**Step 2: draw a boundary by eye.** Scan the visits column. Everyone with more than 9 visits stayed, except C019. Everyone with 9 or fewer churned, except C017 and C020. So the rule *"visits > 9 → stays, otherwise churns"* gets 17 of 20 right: **85%**.

That vertical line at visits = 9 is a decision boundary. You just built a classifier by hand. The three customers it gets wrong (C017, C019, C020) are the interesting ones: a better boundary would have to bend around them, and in later lessons you will judge whether bending is worth it.

**Step 3: why not a line of best fit?** In the notebook you fit linear regression to the 0/1 column. It produces predictions like **−0.15** and **1.11**. A customer cannot be −15% churned. The line has no idea the target is a category, and its errors push it around in ways that have nothing to do with the boundary. Lesson 03.4 fixes exactly this by bending the line into an S-curve.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Load the 20 customers and plot them coloured by class.
2. Compute the baseline.
3. Write the visits > 9 rule as a function and score it (85%).
4. Fit `LinearRegression` on the 0/1 target and look at the nonsense predictions.
5. Fit your first real classifier, `KNeighborsClassifier`, and draw its decision boundary. (Lesson 03.2 explains how it works; here you only look.)

The pattern of every scikit-learn classifier is identical to regression:

```python
model.fit(X_train, y_train)      # learn
model.predict(X_new)             # → array of 0/1 labels
model.score(X_test, y_test)      # → accuracy
```

## 7. On real data

The notebook rebuilds the customer table from lesson 01.9 (orders before 1 Sep 2011 → features; no order in Sep–Nov → `churned = 1`): about 3,300 customers, 44% churned, baseline 56%. You compute the baseline, apply a hand rule on `days_since_last`, and see how far a one-line rule gets you (about 65%). That is the number every model in this module must beat.

## 8. Where it breaks

- **Accuracy hides the baseline.** "65% accurate" sounds fine until you learn 56% was free. Always report the baseline next to the score.
- **Imbalanced classes.** With 2% fraud, "nothing is fraud" is 98% accurate and catches no fraud. Lesson 03.6 gives better measures.
- **Boundaries by eye do not scale.** Two features fit on a chart; forty do not. The algorithms in 03.2–03.4 find boundaries in any number of dimensions.
- **The label must be defined before the features.** "Churned" here means "no order in the next 90 days", measured after a fixed cut-off date. Define it sloppily and you will leak the answer into the features (Module 04).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Find a *two-part* hand rule (using both visits and days since last order) that beats 85% on the 20 customers. Write it as a function and score it.
2. On the real table, try thresholds of 30, 60, 90, 120 and 180 days for the rule "days_since_last > T → churn". Which T is best? Plot accuracy against T.
3. Change the churn definition to "no order in the next **60** days" (future window ends 2011-10-31). How do the churn rate and baseline change?

**Quiz**

1. What is the difference between a regression target and a classification target?
2. 30% of emails are spam. What accuracy does the majority-class baseline get, and what does it predict?
3. What is a decision boundary?
4. Why do predictions like −0.15 or 1.11 show that linear regression is the wrong tool here?
5. Three features are easy to draw; forty are not. What does an algorithm do that your eye cannot?

## 10. Recap

- Classification predicts a **class**; the class you are hunting for is coded 1 (positive).
- A classifier draws a **decision boundary** in feature space; every algorithm is a different recipe for it.
- Always compute the **majority-class baseline** first; on the 20 customers it is 55%, on the real table 56%.
- A single hand-drawn boundary (visits > 9) already scores 85% on 20 rows; algorithms find such boundaries in any number of dimensions.
- Linear regression on a 0/1 target produces impossible values; classification needs its own models.

**Next:** [03.2 K-nearest neighbours by hand](../02-knn-by-hand/): the simplest classifier there is, and you compute every step of it.
