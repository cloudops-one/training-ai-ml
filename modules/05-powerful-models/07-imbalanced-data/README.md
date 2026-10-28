# 05.7 · Imbalanced data: class weights, resampling, SMOTE, the right metrics

⏱ about 1.5 hours · Prerequisites: 03.6–03.7 (metrics, thresholds), 04.8 (pipelines) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's VIP team plans a preview event for next quarter's biggest spenders: customers who will spend more than £2,393 between September and November. That is the top 5%: 166 of 3,317 customers.

You train a classifier and proudly report **95.8% accuracy**. Then someone points out that a "model" which says *nobody* is a VIP scores **94.9%**, and finds zero VIPs. Your real model, at the default threshold, flags 19 customers and finds only 13 of the 42 VIPs in the test set.

Fraud, defaults, rare diseases, machine failures, VIP customers: the cases that matter most are usually rare. This lesson is about modelling them without being fooled.

## 2. The idea

When one class is 95% of the data, a model can get a low loss by mostly ignoring the other 5%. Two things go wrong:

1. **The metric lies.** Accuracy is dominated by the common class.
2. **The model is timid.** It rarely gives a rare-class probability above 0.5, so at the default threshold it catches few rare cases.

The fixes all aim at the same goal, making the rare class "count" more:

- **Measure differently**: precision, recall, F1, PR-AUC instead of accuracy.
- **Weight differently**: make a mistake on a rare example cost more (class weights).
- **Sample differently**: copy rare rows, drop common rows, or synthesise new rare rows (SMOTE).
- **Decide differently**: lower the threshold.

## 3. Definition

- **Class imbalance**: one class is much rarer than the other(s). Rough guide: below 10% is imbalanced; below 1% is severely imbalanced.
- **Positive rate**: share of the rare (positive) class, here 5%.
- **Class weight**: a multiplier on each example's contribution to the loss. `"balanced"` gives each class the same total weight.
- **Random over-sampling**: duplicate rare examples until classes are equal in the training data.
- **Random under-sampling**: discard common examples until classes are equal.
- **SMOTE** (Synthetic Minority Over-sampling TEchnique): create new rare examples on the lines between existing rare examples and their nearest rare neighbours.
- **PR-AUC** (average precision): area under the precision–recall curve. Its random baseline equals the positive rate.

## 4. Formula

**Balanced class weight** for class $c$:

$$w_c = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_c}$$

**SMOTE** synthetic point, for a rare point $x$, one of its $k$ nearest rare neighbours $x_{nn}$, and a random $\lambda \in [0, 1]$:

$$x_{\text{new}} = x + \lambda\,(x_{nn} - x)$$

**Random baselines** of the two curve summaries:

$$\text{ROC-AUC}_{\text{random}} = 0.5 \qquad \text{PR-AUC}_{\text{random}} = \text{positive rate}$$

## 5. By hand

**a) The lying model.** Test set: 830 customers, 42 VIPs. Predict "nobody":

- Accuracy = (830 − 42) / 830 = 788 / 830 = **0.949**
- Recall = 0 / 42 = **0**
- Precision: undefined (no positive predictions), reported as 0.

**b) Class weights.** Training set: 2,487 customers, 124 VIPs, 2,363 others.

$$w_{\text{VIP}} = \frac{2487}{2 \times 124} = \frac{2487}{248} = 10.03 \qquad w_{\text{other}} = \frac{2487}{2 \times 2363} = \frac{2487}{4726} = 0.526$$

Check: $124 \times 10.03 \approx 1244$ and $2363 \times 0.526 \approx 1243$. Both classes now carry the same total weight. One misclassified VIP costs as much as 19 misclassified ordinary customers.

**c) One SMOTE point.** Rare customer $x = (2, 3)$ in scaled (orders, spend). Its nearest rare neighbour $x_{nn} = (4, 7)$. Draw $\lambda = 0.25$:

$$x_{\text{new}} = (2, 3) + 0.25 \times \big((4, 7) - (2, 3)\big) = (2, 3) + 0.25 \times (2, 4) = (2, 3) + (0.5, 1) = (2.5, 4)$$

A quarter of the way from $x$ to its neighbour. SMOTE never invents points outside the space between real rare points.

## 6. In code

```python
# class weights: one argument
LogisticRegression(class_weight="balanced")

# resampling: use imblearn's Pipeline so it only happens on training folds
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
model = make_pipeline(StandardScaler(), SMOTE(random_state=42), LogisticRegression())

# metrics that respect rarity
from sklearn.metrics import average_precision_score, classification_report
average_precision_score(y_test, proba)          # PR-AUC
```

Choosing a threshold honestly, from cross-validated training predictions:

```python
p_cv = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, p_cv)
best_t = thr[(2 * prec * rec / (prec + rec))[:-1].argmax()]
```

## 7. On real data

The test-set comparison (logistic regression, threshold 0.5 unless stated):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| predict nobody | 0.949 | 0 | 0 | 0 | 0.50 | 0.051 |
| plain | 0.958 | 0.684 | 0.310 | 0.426 | 0.919 | 0.525 |
| class_weight="balanced" | 0.901 | 0.318 | 0.833 | 0.461 | 0.940 | 0.521 |
| random oversample | 0.900 | 0.315 | 0.833 | 0.458 | 0.946 | 0.524 |
| random undersample | 0.864 | 0.255 | 0.881 | 0.396 | 0.927 | 0.542 |
| SMOTE | 0.896 | 0.307 | 0.833 | 0.449 | 0.939 | 0.500 |
| **plain, threshold 0.24** | 0.959 | 0.611 | 0.524 | **0.564** | 0.919 | 0.525 |

What this shows:

- Every rebalancing fix trades precision for recall in roughly the same way. None is magic.
- PR-AUC, which measures how well the model *ranks* VIPs, hardly changes. The fixes mostly move the decision line.
- The best F1 comes from the **plain model with a threshold chosen on training CV** (0.24). Free, simple, honest.

And the leak, measured by cross-validated PR-AUC with a random forest:

| | CV PR-AUC |
|---|---|
| SMOTE applied **before** CV (wrong) | 0.992 |
| SMOTE **inside** an imblearn pipeline (right) | 0.579 |

The 0.992 is fiction.

## 8. Where it breaks

- **Resampling before splitting or before CV.** The leak above. Always resample inside a pipeline, only on training folds.
- **Resampling the test set.** Never. The test set must look like the real world, imbalance included.
- **Trusting probabilities after rebalancing.** Weighting or resampling inflates rare-class probabilities. A predicted 0.6 no longer means "60% chance". If you need calibrated probabilities (e.g. for expected-value calculations), recalibrate or use the plain model with a threshold.
- **Under-sampling throws away data.** Here it trained on 248 customers out of 2,487. Fine for huge datasets; wasteful for small ones.
- **SMOTE on categories or mixed data.** Interpolating between "UK" and "France" means nothing. Use `SMOTENC` for mixed data, or prefer class weights.
- **Optimising F1 blindly.** F1 weighs precision and recall equally. If a missed fraud costs £500 and a false alarm £2, use expected cost (03.7) instead.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: 20,000 orders, 60 fraud. Accuracy and recall of "nothing is fraud"; random PR-AUC; balanced class weights.
2. On paper: SMOTE points between (10, 2) and (14, 1) for λ = 0.1, 0.5, 0.9.
3. Make the target rarer (top 2%). What happens to recall at 0.5 and to PR-AUC?
4. Repeat the comparison with LightGBM. Which fix helps most?

**Quiz**

1. Why can a useless model have 95% accuracy?
2. 1,000 training rows, 40 positive. What are the balanced class weights?
3. SMOTE with $x = (1, 1)$, $x_{nn} = (3, 5)$, $\lambda = 0.5$. What is $x_{\text{new}}$?
4. Why is SMOTE before cross-validation a leak?
5. On a 2%-positive problem, which is more informative: ROC-AUC or PR-AUC? Why?

## 10. Recap

- On rare targets, **accuracy lies**. Report precision, recall, F1 and **PR-AUC** (whose random baseline is the positive rate).
- **Class weights** make rare mistakes cost more: $w_c = n / (k \cdot n_c)$.
- **Over-sampling, under-sampling and SMOTE** change the training rows; SMOTE adds points between rare neighbours: $x + \lambda(x_{nn} - x)$.
- Resample **only inside training folds**, using an imblearn Pipeline. Never touch the test set.
- Most fixes move the threshold rather than improve ranking. **Try choosing the threshold first.**

**Next:** [05.8 Explaining models](../08-explaining-models/) answers the CEO's question: *why* does the model say this customer will churn?
