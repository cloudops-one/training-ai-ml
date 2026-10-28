# 03.7 · ROC curves, AUC and choosing a threshold by business cost

⏱ about 1.5 hours · Prerequisites: 03.4 (predict_proba), 03.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing now understands precision and recall. Their next question is sharper: *"The model gives each customer a churn probability. Above what probability should we send the £5 voucher?"*

In 03.6 you used scikit-learn's default: flag churn when the probability is **≥ 0.5**. Nothing about ShopWave's business went into that number. If a lost customer costs £60 and a voucher £5, maybe it is worth sending vouchers at 0.2. If vouchers were £50, maybe only at 0.8.

The **threshold** is a business decision disguised as a default. This lesson makes it explicit.

## 2. The idea

A classifier that outputs probabilities is really a *family* of classifiers, one per threshold. Slide the threshold down and you flag more people: recall goes up, precision goes down, false alarms multiply. Slide it up and the opposite happens.

Two tools make this visible:

- The **ROC curve** plots, for every threshold, how many real churners you catch (true positive rate) against how many stayers you wrongly flag (false positive rate). A perfect model hugs the top-left corner; a coin-flip is the diagonal. The **area under it (AUC)** summarises the model's ranking quality *independent of any threshold*.
- The **expected cost** at each threshold: (number of vouchers × £5) + (number of missed churners × £60). The best threshold is simply the cheapest one.

## 3. Definition

For a given threshold $t$, predict churn when $P(\text{churn}) \ge t$, then count TP, FP, FN, TN as in 03.6.

- **True positive rate (TPR)** = recall = TP / (TP + FN). Fraction of churners caught.
- **False positive rate (FPR)** = FP / (FP + TN). Fraction of stayers wrongly flagged.
- **ROC curve**: the points (FPR, TPR) for all thresholds from 1 down to 0, joined up.
- **AUC** (area under the ROC curve): a number from 0.5 (random) to 1.0 (perfect). It equals the probability that a randomly chosen churner is scored *higher* than a randomly chosen stayer.
- **Precision–recall curve**: precision against recall for all thresholds. More informative than ROC when positives are rare.

## 4. Formula

$$\text{TPR}(t) = \frac{TP_t}{TP_t + FN_t} \qquad \text{FPR}(t) = \frac{FP_t}{FP_t + TN_t}$$

$$\text{AUC} = \frac{\#\{(\text{churner}, \text{stayer}) \text{ pairs with } p_{\text{churner}} > p_{\text{stayer}}\}}{\#\text{churners} \times \#\text{stayers}}$$

$$\text{Cost}(t) = c_{\text{voucher}} \cdot (TP_t + FP_t) + c_{\text{lost}} \cdot FN_t$$

with $c_{\text{voucher}} = £5$ and $c_{\text{lost}} = £60$ for ShopWave. (A voucher to a true churner is still a cost, but we assume it *keeps* them, so the £60 is not incurred.)

## 5. By hand

Ten customers, scored by the model, sorted by probability. `y` = 1 if they really churned. Five churners, five stayers.

| customer | $P(\text{churn})$ | $y$ |
|---|---|---|
| A | 0.95 | 1 |
| B | 0.85 | 1 |
| C | 0.80 | 0 |
| D | 0.70 | 1 |
| E | 0.60 | 0 |
| F | 0.55 | 1 |
| G | 0.40 | 0 |
| H | 0.30 | 1 |
| I | 0.20 | 0 |
| J | 0.10 | 0 |

**Three thresholds.** Flag everyone at or above $t$; count.

| $t$ | flagged | TP | FP | FN | TN | TPR | FPR | precision |
|---|---|---|---|---|---|---|---|---|
| 0.90 | A | 1 | 0 | 4 | 5 | 1/5 = 0.2 | 0/5 = 0.0 | 1/1 = 1.00 |
| 0.50 | A–F | 4 | 2 | 1 | 3 | 4/5 = 0.8 | 2/5 = 0.4 | 4/6 = 0.67 |
| 0.25 | A–H | 5 | 3 | 0 | 2 | 5/5 = 1.0 | 3/5 = 0.6 | 5/8 = 0.63 |

Those three (FPR, TPR) points, (0, 0.2), (0.4, 0.8), (0.6, 1.0), plus (0, 0) and (1, 1), are five points on the ROC curve. The notebook draws all of them.

**AUC by counting pairs.** For each churner, count the stayers scored *below* them:

- A (0.95): beats C, E, G, I, J → 5
- B (0.85): beats C, E, G, I, J → 5
- D (0.70): beats E, G, I, J → 4
- F (0.55): beats G, I, J → 3
- H (0.30): beats I, J → 2

Total 19 of 5 × 5 = 25 pairs. **AUC = 19/25 = 0.76.** Pick a random churner and a random stayer; 76% of the time the model ranks them correctly.

**Cost at each threshold** (£5 per voucher sent, £60 per missed churner):

| $t$ | vouchers (TP + FP) | missed (FN) | cost |
|---|---|---|---|
| 0.90 | 1 | 4 | 5 × 1 + 60 × 4 = **£245** |
| 0.50 | 6 | 1 | 5 × 6 + 60 × 1 = **£90** |
| 0.25 | 8 | 0 | 5 × 8 + 60 × 0 = **£40** |
| 0.00 (everyone) | 10 | 0 | 5 × 10 = **£50** |

The default 0.5 costs more than twice as much as 0.25. And blindly sending vouchers to everyone (£50) is *also* better than the default. The model earns its keep only when the threshold is chosen with the costs in view.

## 6. In code

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

fpr, tpr, thresholds = roc_curve(y, p)        # every threshold at once
auc = roc_auc_score(y, p)                     # 0.76 for the ten customers

def cost(y, p, t, c_voucher=5, c_lost=60):
    pred = (p >= t).astype(int)
    TP = ((y == 1) & (pred == 1)).sum(); FP = ((y == 0) & (pred == 1)).sum(); FN = ((y == 1) & (pred == 0)).sum()
    return c_voucher * (TP + FP) + c_lost * FN
```

The notebook reproduces the table above, draws the ROC and precision–recall curves for the ten customers, and plots cost against threshold.

## 7. On real data

On the ShopWave customer table: the 03.4 logistic regression's ROC curve and AUC on the test customers, the precision–recall curve, and a cost-versus-threshold plot using £5 / £60. You will read off the cheapest threshold and compare its total cost with the default 0.5 and with "voucher for everyone". You will also change the cost of a lost customer and watch the best threshold move.

## 8. Where it breaks

- **AUC ignores the threshold entirely.** Two models with the same AUC can behave very differently at the threshold you actually deploy. AUC is for comparing models' ranking ability, not for deciding what to ship.
- **ROC looks optimistic on rare positives.** With 2% fraud, FPR stays tiny even when you flag thousands of innocent transactions. Use the precision–recall curve and **average precision** when positives are rare.
- **The costs are guesses.** £60 per lost customer is an estimate; if it is really £20, the best threshold changes. Run the cost curve for a range of values and see how sensitive the decision is.
- **Probabilities may not be calibrated.** A model that says 0.7 is not always right 70% of the time. Tree ensembles in particular are often over- or under-confident; the threshold still works, but "0.7" should not be read as a literal probability without checking.
- **Thresholds chosen on the test set are optimistic.** Choose on validation data, confirm on test (Module 04).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: add threshold $t = 0.65$ to the table (TP, FP, FN, TN, TPR, FPR, cost). Where does it sit on the ROC curve?
2. By hand: swap the scores of customers C and H (so C = 0.30, H = 0.80). Recount the pairs. What is the new AUC?
3. In code: the voucher now costs £20. Recompute the cost table for the ten customers. Which threshold is cheapest now?
4. In code: on the real test set, plot cost against threshold for lost-customer costs of £20, £60 and £150 on the same chart. How does the best threshold move, and why?

**Quiz**

1. What two quantities does an ROC curve plot?
2. A model has AUC 0.5. What does that say about it?
3. Why can two models with identical AUC give different precision at threshold 0.5?
4. When is the precision–recall curve a better picture than ROC?
5. The default threshold of 0.5 is "correct" only under one assumption about costs. Which?

## 10. Recap

- A probability model is a family of classifiers; the **threshold** picks one.
- **ROC curve** = TPR against FPR for all thresholds; **AUC** = probability the model ranks a random positive above a random negative (0.5 random, 1.0 perfect).
- **AUC compares models; cost picks the threshold.** Cost($t$) = £5 × flagged + £60 × missed; choose the cheapest $t$.
- The default 0.5 assumes both mistakes cost the same. At ShopWave they do not, and 0.25 is 2× cheaper.
- When positives are rare, prefer the precision–recall curve.

**Next:** [03.8 Multi-class classification](../08-multiclass/) extends everything so far from two classes to many.
