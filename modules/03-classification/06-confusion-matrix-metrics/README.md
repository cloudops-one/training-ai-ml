# 03.6 · The confusion matrix: accuracy, precision, recall, F1

⏱ about 2 hours · Prerequisites: 03.1–03.4 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Your churn model from 03.4 reports **95% accuracy**. Marketing is delighted. Then they send a £5 retention voucher to everyone the model flagged, and three months later just as many customers have left as before.

What went wrong? Only 5% of ShopWave's customers churn in a given quarter. A model that says *"nobody churns"* is right 95% of the time and catches **zero** churners. Accuracy did not measure the thing marketing cared about: *of the people who were going to leave, how many did we catch?*

One number cannot describe a classifier. You need four, and a table to hold them.

## 2. The idea

Every prediction lands in one of four boxes. Take "churn" as the **positive** class (the thing we are trying to detect):

```
                          predicted churn      predicted stay
  actually churned     |  True Positive  TP  |  False Negative FN  |   ← churners we caught / missed
  actually stayed      |  False Positive FP  |  True Negative  TN  |   ← false alarms / correct all-clears
```

Two kinds of mistake, and they cost different things:

- **False negative** (missed churner): the customer leaves and we did nothing. Cost: a lost customer, say £60 of future profit.
- **False positive** (false alarm): we send a voucher to someone who was staying anyway. Cost: £5.

A doctor screening for a disease, a bank flagging fraud, a spam filter: every classifier has these two mistakes, and which one hurts more is a business question, not a maths question. The metrics below let you *see* the trade-off instead of hiding it behind one number.

## 3. Definition

The **confusion matrix** is the 2×2 table of counts above (scikit-learn prints it with actual classes as rows, predicted as columns, and class 0 first).

From its four cells:

- **Accuracy**: fraction of all predictions that were right.
- **Precision**: of the customers we *flagged* as churn, the fraction that really churned. "When the model cries wolf, how often is there a wolf?"
- **Recall** (also *sensitivity*, *true positive rate*): of the customers who *really* churned, the fraction we flagged. "Of all the wolves, how many did we spot?"
- **F1 score**: a single number that is high only when precision *and* recall are both high. It is the *harmonic* mean, which punishes imbalance: precision 1.0 with recall 0.1 gives F1 = 0.18, not 0.55.

## 4. Formula

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \qquad\qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

Read the denominators: precision divides by the **predicted-positive column** (TP + FP), recall divides by the **actually-positive row** (TP + FN). That is the whole difference between them.

## 5. By hand

Take the 20 ShopWave customers in `shopwave_customers_small.csv`, where `churned` is known. Use the simplest possible classifier, a rule: **predict churn if `days_since_last` > 30.**

| customer | days_since_last | predicted | churned | box |
|---|---|---|---|---|
| C001 | 55 | 1 | 1 | TP |
| C002 | 44 | 1 | 0 | FP |
| C003 | 22 | 0 | 0 | TN |
| C004 | 58 | 1 | 1 | TP |
| C005 | 25 | 0 | 1 | FN |
| C006 | 20 | 0 | 0 | TN |
| C007 | 54 | 1 | 1 | TP |
| C008 | 22 | 0 | 0 | TN |
| C009 | 5 | 0 | 1 | FN |
| C010 | 28 | 0 | 1 | FN |
| C011 | 47 | 1 | 1 | TP |
| C012 | 12 | 0 | 0 | TN |
| C013 | 28 | 0 | 0 | TN |
| C014 | 8 | 0 | 0 | TN |
| C015 | 41 | 1 | 0 | FP |
| C016 | 29 | 0 | 0 | TN |
| C017 | 20 | 0 | 0 | TN |
| C018 | 14 | 0 | 1 | FN |
| C019 | 34 | 1 | 1 | TP |
| C020 | 40 | 1 | 0 | FP |

Count the boxes: **TP = 5, FP = 3, FN = 4, TN = 8.** (Check: 5 + 3 + 4 + 8 = 20.)

```
                   predicted churn   predicted stay
 actually churned        5                4          (9 churners)
 actually stayed         3                8          (11 stayers)
```

Now the four numbers:

- Accuracy = (5 + 8) / 20 = 13 / 20 = **0.65**
- Precision = 5 / (5 + 3) = 5 / 8 = **0.625**. Five of the eight vouchers went to real churners.
- Recall = 5 / (5 + 4) = 5 / 9 = **0.556**. We caught five of the nine churners and missed four.
- F1 = 2 × 0.625 × 0.556 / (0.625 + 0.556) = 0.694 / 1.181 = **0.588**

Compare with the lazy baseline **"nobody churns"**: TP = 0, FP = 0, FN = 9, TN = 11. Accuracy = 11/20 = 0.55, recall = 0, precision undefined (0/0). Our rule is better on accuracy by only 10 points, but on recall it is infinitely better: it actually finds churners.

**The 95% trap, by hand.** 1,000 customers, 50 churn. "Nobody churns": TP = 0, FN = 50, TN = 950, FP = 0. Accuracy = 950/1000 = **95%**. Recall = 0/50 = **0%**. That is the model marketing was delighted with.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The same 20 rows, with the rule as a one-liner, then scikit-learn's functions, which must reproduce your numbers exactly:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

pred = (cust["days_since_last"] > 30).astype(int)
print(confusion_matrix(cust["churned"], pred))   # [[TN FP] [FN TP]]  →  [[8 3] [4 5]]
print(accuracy_score(cust["churned"], pred))     # 0.65
print(precision_score(cust["churned"], pred))    # 0.625
print(recall_score(cust["churned"], pred))       # 0.556
print(f1_score(cust["churned"], pred))           # 0.588
```

Note scikit-learn's layout: row 0 is *actual stay*, row 1 is *actual churn*, so TP is bottom-right. Then `classification_report`, which prints precision, recall and F1 for **both** classes at once, plus two averages you will meet again in 03.8.

## 7. On real data

The notebook rebuilds the ShopWave customer table from 01.9 (features before 1 Sep 2011, target: did they buy again by 1 Dec 2011) and trains the logistic regression from 03.4. You compute its confusion matrix and all four metrics on the held-out customers, compare three candidates (the "nobody churns" baseline, a recency rule, the model) in one table, and see why the "best" one depends on which metric you read.

## 8. Where it breaks

- **Accuracy on imbalanced classes.** The whole point of this lesson. If one class is rare, accuracy is almost meaningless; report precision and recall.
- **Precision alone** can be gamed by predicting positive only when certain (one correct flag → precision 1.0, recall ≈ 0). **Recall alone** can be gamed by flagging everyone (recall 1.0, precision = base rate). Always read them together, or use F1.
- **F1 treats both mistakes as equally bad.** If a missed churner costs £60 and a wasted voucher £5, they are not. Lesson 03.7 replaces F1 with the actual cost.
- **Which class is "positive"?** Precision and recall change if you swap. Say which class you mean. In scikit-learn, `pos_label=1` by default.
- **Metrics on the training rows flatter the model.** Always compute them on data the model did not see (03.1, and properly in Module 04).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: change the rule to `days_since_last > 40`. Fill the four boxes and compute all four metrics. Did precision go up or down? Recall?
2. By hand: a fraud model on 10,000 transactions flags 120; 80 of them are fraud; there were 200 frauds in total. Write the confusion matrix and compute precision, recall, F1 and accuracy.
3. In code: on the real customer table, find the `days_since_last` threshold (try −1 meaning "flag everyone", 15, 30, 60, 90, 120, 150) with the highest F1. Then the one with the highest recall. Why are they different?
4. In code: compute precision and recall for the *stay* class (`pos_label=0`) for the 03.4 model. Why does `classification_report` show both?

**Quiz**

1. A model has precision 0.90 and recall 0.10. Describe its behaviour in one sentence.
2. Which cell of the confusion matrix do precision and recall share in the numerator?
3. 2% of emails are spam. A filter marks nothing as spam. What is its accuracy and its recall?
4. Why is F1 the harmonic mean rather than the ordinary average?
5. Marketing says "we cannot afford to miss churners". Which metric should the model be tuned for?

## 10. Recap

- Every prediction is a TP, FP, FN or TN; the **confusion matrix** counts them.
- **Accuracy** = (TP + TN) / all. It lies when classes are imbalanced.
- **Precision** = TP / (TP + FP): how trustworthy a positive flag is. **Recall** = TP / (TP + FN): how many positives we caught.
- **F1** combines them with a harmonic mean; it is low if either is low.
- Always compare against the "predict the majority class" baseline, and always say which class is positive.

**Next:** [03.7 ROC curves, AUC and choosing a threshold by business cost](../07-roc-auc-threshold/) shows that precision and recall are not fixed: they move as you change the probability cut-off, and the right cut-off is the one that costs least.
