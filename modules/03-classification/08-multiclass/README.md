# 03.8 · Multi-class classification

⏱ about 1 hour · Prerequisites: 03.4, 03.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing has three catalogues, not one: the **premium** catalogue (expensive to print, for customers expected to spend over £500 next quarter), the **standard** catalogue (£1–£500), and **none** (customers expected not to buy at all; send an email instead). Every customer must get exactly one.

So far every classifier has answered yes/no. Now the answer is one of three. Everything you learned still applies, with two small extensions: how a model picks among several classes, and how precision and recall work when there is no single "positive" class.

## 2. The idea

Two ways to turn yes/no classifiers into a many-class one:

- **One-vs-rest**: train three yes/no models ("premium or not?", "standard or not?", "none or not?"), score a customer with all three, pick the highest. Any binary model can be used this way.
- **One model, several outputs**: logistic regression, trees, KNN and neural networks can natively produce one score per class. For logistic regression the scores are turned into probabilities that sum to 1 by the **softmax** function, which you will meet again in every neural network in Module 08.

For evaluation, the confusion matrix simply grows to 3×3, and precision/recall are computed **per class** (treating that class as "positive" and the other two as "negative"), then averaged.

## 3. Definition

- **Multi-class classification**: predicting exactly one of $K > 2$ labels.
- **One-vs-rest (OvR)**: $K$ binary classifiers; predict the class whose classifier is most confident.
- **Softmax**: turns $K$ raw scores $z_1, \dots, z_K$ into probabilities that are positive and sum to 1. The largest score gets the largest probability; differences are exaggerated exponentially.
- **Macro average**: average the per-class metric giving each class equal weight, however rare.
- **Weighted average**: average weighted by each class's support (number of true rows), so big classes dominate.
- **Micro average**: pool all TP/FP/FN across classes first; for single-label multi-class this equals accuracy.

## 4. Formula

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Per-class metrics from a $K \times K$ confusion matrix $M$ (rows = actual, columns = predicted):

$$\text{precision}_k = \frac{M_{kk}}{\sum_i M_{ik}} \;(\text{column } k) \qquad \text{recall}_k = \frac{M_{kk}}{\sum_j M_{kj}} \;(\text{row } k)$$

$$\text{macro-}F_1 = \frac{1}{K}\sum_k F_{1,k} \qquad \text{weighted-}F_1 = \sum_k \frac{n_k}{n} F_{1,k}$$

## 5. By hand

**Softmax on three scores.** A model gives one customer raw scores $z = (2, 1, 0)$ for (premium, standard, none).

$e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$; sum $= 11.107$.

$$\text{softmax} = \left(\frac{7.389}{11.107}, \frac{2.718}{11.107}, \frac{1}{11.107}\right) = (0.665, 0.245, 0.090)$$

Predict **premium** with 66.5% probability. Note that a difference of 1 in score became a factor of $e \approx 2.7$ in probability.

**A 3×3 confusion matrix.** Nine customers, actual vs predicted catalogue:

| actual \ predicted | premium | standard | none | row total |
|---|---|---|---|---|
| **premium** | 2 | 1 | 0 | 3 |
| **standard** | 1 | 2 | 1 | 4 |
| **none** | 0 | 0 | 2 | 2 |
| column total | 3 | 3 | 3 | 9 |

Diagonal = correct: $2 + 2 + 2 = 6$. **Accuracy = 6/9 = 0.667.**

Per class (precision uses the column, recall uses the row):

| class | precision | recall | $F_1$ | support |
|---|---|---|---|---|
| premium | 2/3 = 0.667 | 2/3 = 0.667 | 0.667 | 3 |
| standard | 2/3 = 0.667 | 2/4 = 0.500 | 0.571 | 4 |
| none | 2/3 = 0.667 | 2/2 = 1.000 | 0.800 | 2 |

- **Macro-$F_1$** = (0.667 + 0.571 + 0.800) / 3 = **0.679**
- **Weighted-$F_1$** = (3 × 0.667 + 4 × 0.571 + 2 × 0.800) / 9 = (2.0 + 2.286 + 1.6) / 9 = **0.654**

Weighted is lower because the biggest class (standard) has the worst $F_1$. If the *rare* class had been the bad one, macro would be lower instead. Macro is the honest number when you care about rare classes equally; weighted tracks "how many customers were served well".

## 6. In code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

clf = LogisticRegression(max_iter=1000)       # multi-class with softmax, automatically
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))   # per-class rows + macro + weighted
```

The notebook verifies the softmax arithmetic, rebuilds the 3×3 table and its metrics, then uses scikit-learn's classic **Iris** dataset (three flower species, four measurements) as a clean warm-up before ShopWave.

## 7. On real data

The ShopWave customer table from 01.9 with a three-way target built from `spend_next_quarter`: **none** (£0), **standard** (£0–£500), **premium** (over £500). Logistic regression and KNN, 3×3 confusion matrix, per-class report, macro vs weighted F1, and a look at which classes get confused with which (hint: the middle class, *standard*, is the hard one; the model keeps mistaking modest future spenders for customers who will not return).

## 8. Where it breaks

- **Macro vs weighted disagree** whenever classes differ in size and quality. Report both and say which you optimised.
- **Ordered classes.** "none < standard < premium" has an order; plain multi-class ignores it, so predicting *none* for a *premium* customer counts the same as predicting *standard*. If the order matters, consider regression on spend (Module 02) followed by cutting into bands, or ordinal methods.
- **Many classes with few examples each** (hundreds of product categories): per-class metrics become noisy; look at top-k accuracy or group rare classes.
- **Multi-*label*** is different: one email can be *both* a complaint *and* a delivery question. That needs one binary classifier per label, not softmax.
- **Thresholds** (03.7) are no longer a single number; with $K$ classes you adjust class weights or costs instead.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: softmax of $z = (3, 3, 0)$. What happens to the first two probabilities, and why is the third so small?
2. By hand: in the 3×3 table, move one customer from (standard → predicted none) to (standard → predicted standard). Recompute accuracy, the standard-class $F_1$, macro and weighted $F_1$.
3. In code: on Iris, fit KNN with k = 1, 5, 15 and print macro-$F_1$ for each. Which species is confused with which?
4. In code: on the ShopWave three-class table, compare `LogisticRegression` with `class_weight="balanced"` against the default. Which per-class recall changes most, and in which direction?

**Quiz**

1. What does softmax guarantee about its outputs?
2. In a 3×3 confusion matrix, where do you read the precision of class 2?
3. When would macro-$F_1$ be much lower than weighted-$F_1$?
4. One-vs-rest trains how many models for 5 classes?
5. Is "tag each product photo with every applicable category" multi-class or multi-label?

## 10. Recap

- Multi-class = pick one of $K$ labels; **one-vs-rest** or a native multi-output model with **softmax**.
- Softmax: $e^{z_k} / \sum_j e^{z_j}$; positive, sums to 1, exaggerates score gaps.
- The confusion matrix becomes $K \times K$; precision reads down a column, recall along a row.
- **Macro** average treats every class equally; **weighted** follows class size; **micro** equals accuracy.
- Say which average you report, and whether the classes have an order.

**Next:** [03.P Mini-project: Titanic survival, then ShopWave churn](../project/) puts the whole module to work.
