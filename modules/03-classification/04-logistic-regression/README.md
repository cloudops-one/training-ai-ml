# 03.4 · Logistic regression: the sigmoid and probabilities

⏱ about 2 hours · Prerequisites: 03.1, 02.2, 02.5 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention has 500 vouchers and 3,300 customers. "Churn / stay" is not enough; they need a **ranking**: who is *most* likely to leave, so the top 500 get a voucher. That needs a **probability** for each customer, and a model whose probabilities you can trust and explain.

In 03.1 you saw linear regression produce −0.15 and 1.11 for a 0/1 target. The fix is not to abandon the line. It is to **bend it**.

## 2. The idea

Keep the linear model from Module 02, $z = c + m_1 x_1 + m_2 x_2$. It produces any number from $-\infty$ to $+\infty$. Now pass $z$ through a function shaped like an S that squashes everything into the range 0 to 1:

```
p
 1.0 |                       ________
     |                   ,--'
 0.5 |- - - - - - - - -/- - - - - - - -
     |             ,--'
 0.0 |____________'
     +------------+---------------------  z
                 z = 0
```

Large positive $z$ → probability near 1. Large negative $z$ → near 0. $z = 0$ → exactly 0.5. That S is the **sigmoid**, and a linear model followed by a sigmoid is **logistic regression**. Despite the name, it is a classifier, and it is the most used classifier in business because it is fast, stable, and every coefficient has a meaning.

## 3. Definition

**Logistic regression** models the probability of the positive class as the sigmoid of a linear combination of the features:

$$\hat{p} = \sigma(z), \qquad z = c + m_1 x_1 + \dots + m_p x_p$$

- $z$: the **logit** or **log-odds**. The linear part, exactly as in Module 02.
- $\sigma$: the **sigmoid** function.
- $\hat{p}$: predicted probability of class 1.
- **Decision threshold**: predict 1 if $\hat{p} > 0.5$ (equivalently $z > 0$). Lesson 03.7 shows this 0.5 is a choice, not a law.
- Training finds the $m$'s and $c$ that make the predicted probabilities match the observed 0/1 labels as closely as possible, measured by **log loss**.

## 4. Formula

**Sigmoid:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Log loss** (the cost function training minimises), for one row with true label $y$ and prediction $\hat{p}$:

$$L = -\big[\, y \ln \hat{p} + (1 - y) \ln(1 - \hat{p}) \,\big]$$

Read it as two cases. If $y = 1$ the loss is $-\ln \hat{p}$: small when $\hat{p}$ is near 1, huge when near 0. If $y = 0$ it is $-\ln(1 - \hat{p})$: the mirror. Confident and wrong is punished hardest. The total loss is the average over rows; there is no closed-form solution like 02.2, so it is minimised by gradient descent (02.4).

**Coefficients:** $m_j$ is the change in log-odds for a one-unit increase in $x_j$. $e^{m_j}$ is the multiplicative change in **odds** $\big(\hat{p} / (1 - \hat{p})\big)$. Negative $m_j$ → feature lowers churn probability.

## 5. By hand

**Step 1: the sigmoid at three points.**

| $z$ | $e^{-z}$ | $1 + e^{-z}$ | $\sigma(z)$ |
|---|---|---|---|
| −2 | 7.389 | 8.389 | **0.119** |
| 0 | 1 | 2 | **0.500** |
| 2 | 0.135 | 1.135 | **0.881** |

Symmetric around 0.5; $\sigma(-z) = 1 - \sigma(z)$.

**Step 2: a fitted model.** On the 20 customers (features visits, days since last), scikit-learn finds

$$z = 3.13 - 0.60 \times \text{visits} + 0.052 \times \text{days}$$

Each extra visit lowers the log-odds of churn by 0.60; each extra day since the last order raises it by 0.052. Signs match common sense.

**Step 3: predict for two customers.**

C001 (2 visits, 55 days): $z = 3.13 - 1.20 + 2.86 = 4.79$. $\sigma(4.79) = 1 / (1 + e^{-4.79}) = 1 / (1 + 0.0083) = \mathbf{0.99}$. Churn, with confidence.

C012 (14 visits, 12 days): $z = 3.13 - 8.40 + 0.62 = -4.65$. $\sigma(-4.65) = 1 / (1 + 104.6) = \mathbf{0.01}$. Stays.

C005 (7 visits, 25 days): $z = 3.13 - 4.20 + 1.30 = 0.23$. $\sigma(0.23) = \mathbf{0.56}$. Barely churn; this is the customer the voucher budget should reach *last* among churners, and the kind of customer where the threshold matters.

**Step 4: log loss for two predictions.**

C001 is a churner ($y = 1$) with $\hat{p} = 0.99$: $L = -\ln 0.99 = \mathbf{0.010}$. Nearly free.

Suppose a bad model gave C001 $\hat{p} = 0.10$: $L = -\ln 0.10 = \mathbf{2.303}$. Two hundred times worse. Being confidently wrong is what log loss punishes.

**Step 5: odds.** Coefficient for visits is −0.60, so $e^{-0.60} = 0.55$: each extra visit *multiplies the odds of churn by 0.55*, roughly halving them. That is the sentence you say to the retention team.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The sigmoid is one line; the fit is the same three as always:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X20, y20)
logit.coef_, logit.intercept_          # m's and c
logit.predict_proba(X20)[:, 1]         # p̂(churn) for each customer
logit.predict(X20)                     # 1 if p̂ > 0.5
```

`predict_proba` returns two columns, $[\,1 - \hat{p},\ \hat{p}\,]$; column 1 is the churn probability. You will reproduce it by hand with `sigmoid(X20 @ coef + intercept)` and watch the numbers match. The notebook also draws the decision boundary: for logistic regression it is always a **straight line** ($z = 0$), with probability shading on either side.

## 7. On real data

On the churn table (five features, scaled, 25% held out), logistic regression scores about **67–68%** on test customers against a 56% baseline, level with the best KNN and tree, and it hands you the ranking retention asked for: sort customers by `predict_proba` and take the top 500. The scaled coefficients tell the story: `orders` (−0.99) and `products` (−0.45) lower churn, `days_since_last` (+0.44) raises it, `spend` and `tenure` barely matter once the others are known.

Note the scaling. Logistic regression's answer does not depend on scale, but its *training* (gradient descent) converges faster and its coefficients become comparable when features are standardised. Scale by habit.

## 8. Where it breaks

- **The boundary is a straight line.** If churners form a ring in the middle of the feature space, no line separates them. Trees and KNN bend; logistic regression needs you to add features (squares, products, Lesson 02.6).
- **Outliers in features** stretch the linear part, as in 02.2.
- **Perfectly separable data** sends the coefficients to infinity; scikit-learn's default regularisation (`C=1.0`) keeps them finite. Module 04.10 explains regularisation.
- **Correlated features** share a coefficient between them in unpredictable ways; read coefficients cautiously when features overlap.
- **Probabilities need calibration checks.** "0.7" should mean 70% of such customers churn. Logistic regression is usually well calibrated; trees and KNN often are not.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: compute $\sigma(1)$, $\sigma(-1)$ and $\sigma(5)$. Then compute the log loss for a churner predicted at 0.7 and a stayer predicted at 0.7.
2. By hand, then check: using $z = 3.13 - 0.60\,\text{visits} + 0.052\,\text{days}$, find the number of days since last order at which a customer with 8 visits crosses $\hat{p} = 0.5$. (Set $z = 0$ and solve.)
3. In code: fit logistic regression on `days_since_last` alone, on the real table. Plot $\hat{p}$ against days with the true labels as dots. Where is the 0.5 crossing?
4. Rank the real test customers by `predict_proba` and take the top 500. What fraction of them actually churned? Compare with 500 chosen at random.

**Quiz**

1. What does the sigmoid do to the output of a linear model, and why is that needed?
2. $\sigma(0) = ?$ What decision does $z = 0$ correspond to?
3. A churner is predicted at $\hat{p} = 0.01$. Is the log loss small or large? Why?
4. Coefficient of `orders` is −0.99 (scaled). In one sentence, what does that say about churn?
5. Name a shape of data where logistic regression must fail and a tree can succeed.

## 10. Recap

- Logistic regression = linear model + **sigmoid** $\sigma(z) = 1/(1 + e^{-z})$ → a probability between 0 and 1.
- Trained by minimising **log loss**, which punishes confident mistakes most.
- Coefficients are changes in **log-odds**; $e^{m}$ is the multiplier on the odds.
- On 20 customers: $z = 3.13 - 0.60\,\text{visits} + 0.052\,\text{days}$; C001 → 0.99, C012 → 0.01.
- Gives the **ranking** retention needs; threshold 0.5 is a default you will learn to change.

**Next:** [03.5 Naive Bayes](../05-naive-bayes/) (optional), or jump to [03.6 The confusion matrix](../06-confusion-matrix-metrics/) to learn why accuracy is not enough.
