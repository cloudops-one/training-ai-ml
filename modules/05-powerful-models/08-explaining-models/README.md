# 05.8 · Explaining models: feature importance, permutation importance, SHAP

⏱ about 2 hours · Prerequisites: 03.4 (log-odds), 05.2 (random forest), 05.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The churn model is ready. You present it to ShopWave's CEO, and the first question is not about AUC:

> *"Why does it say customer 13747 is about to leave? And what, in general, makes customers leave? If I can't explain it to the retention team, they won't use it."*

The second question comes from compliance: *"Can you show the model isn't relying on something it shouldn't?"*

A model nobody understands does not get used, and a model nobody checks can quietly learn the wrong thing. Explaining models is part of the job, not an optional extra.

## 2. The idea

There are two kinds of question:

- **Global**: across all customers, which features does the model rely on, and in which direction?
- **Local**: for *this* customer, which facts pushed their prediction up or down, and by how much?

Three tools answer them:

- **Permutation importance** (global): scramble one column and see how much worse the model gets. If scrambling it hurts, the model needed it.
- **Partial dependence** (global, direction): vary one feature for everyone and watch the average prediction move.
- **SHAP values** (local, and global when averaged): split each prediction fairly among the features, so that *average prediction + each feature's push = this customer's prediction*.

## 3. Definition

- **Impurity (built-in) importance**: for tree models, the total impurity reduction from splits on each feature, computed on training data. Fast but biased toward features with many distinct values.
- **Permutation importance**: the drop in a chosen test-set score when one feature's column is randomly shuffled, averaged over several shuffles.
- **Partial dependence**: the average prediction as one feature is set to each value in a grid, with all other features left as they are.
- **Shapley value**: from game theory, each player's average marginal contribution over all orders in which players could join.
- **SHAP value**: the Shapley value of a feature for one prediction. **Base value**: the model's average output. Positive SHAP pushes toward the positive class (churn).

## 4. Formula

**Permutation importance** of feature $j$, with score $S$ (e.g. AUC) on test data:

$$I_j = S(\text{original}) - \frac{1}{R}\sum_{r=1}^{R} S(\text{column } j \text{ shuffled, repeat } r)$$

**Shapley value** of feature $j$, with $F$ all features and $v(T)$ the prediction when only the features in $T$ are known:

$$\phi_j = \sum_{T \subseteq F \setminus \{j\}} \frac{|T|!\,(|F|-|T|-1)!}{|F|!}\,\big[v(T \cup \{j\}) - v(T)\big]$$

which is just "average $j$'s contribution over every joining order".

**Additivity** (the property that makes SHAP readable):

$$\text{prediction} = \text{base value} + \sum_j \phi_j$$

## 5. By hand

**a) Permutation importance on six customers.** The "model" is the rule *churn if days_since_last > 60*.

| customer | orders | days_since_last | churned | rule says |
|---|---|---|---|---|
| 1 | 5 | 10 | 0 | 0 |
| 2 | 2 | 20 | 0 | 0 |
| 3 | 8 | 30 | 0 | 0 |
| 4 | 1 | 90 | 1 | 1 |
| 5 | 3 | 100 | 1 | 1 |
| 6 | 1 | 120 | 1 | 1 |

Original accuracy: 6/6 = **1.0**.

Shuffle `orders` (move the last value to the top): [1, 5, 2, 8, 3, 1]. The rule never looks at orders, so accuracy stays **1.0**. Importance = 1.0 − 1.0 = **0**.

Shuffle `days_since_last` the same way: [120, 10, 20, 30, 90, 100]. The rule now says [1, 0, 0, 0, 1, 1] against truth [0, 0, 0, 1, 1, 1]: customers 2, 3, 5, 6 right, 1 and 4 wrong. Accuracy 4/6 = **0.667**. Importance = 1.0 − 0.667 = **0.333**.

**b) Shapley values for two features.** A toy churn-risk model, with two facts about a customer:

| known facts | predicted risk |
|---|---|
| none (average customer) | 10% |
| A: no order in 60 days | 30% |
| B: only one order ever | 20% |
| A and B | 50% |

Two joining orders:

| order | A's contribution | B's contribution |
|---|---|---|
| A then B | 30 − 10 = 20 | 50 − 30 = 20 |
| B then A | 50 − 20 = 30 | 20 − 10 = 10 |
| **average** | **25** | **15** |

Check additivity: 10 + 25 + 15 = **50**, exactly the prediction. A gets more credit because it moves the risk more, whichever order you look at.

## 6. In code

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap

perm = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
PartialDependenceDisplay.from_estimator(model, X_test.astype(float), ["days_since_last", "orders"])

explainer = shap.TreeExplainer(model)
sv = explainer(X_test)
shap.plots.beeswarm(sv)          # global
shap.plots.waterfall(sv[i])      # one customer
```

For LightGBM and XGBoost classifiers, SHAP values are in **log-odds**. Convert the base value with the sigmoid to say "an average customer has a 42% chance".

## 7. On real data

The tuned LightGBM from 05.6 (test AUC 0.739).

**The impurity trap.** Add a column of random numbers and fit a deep random forest. Built-in importance gives the noise **0.175**, more than `orders` (0.096) and `tenure_days` (0.148). Permutation importance gives it **0.001**. Built-in importance measured how useful the column was for *memorising training customers*; permutation importance measures how useful it is on *new* ones.

**What the churn model relies on** (permutation importance, drop in test AUC):

| feature | drop in AUC |
|---|---|
| orders | 0.054 |
| spend | 0.026 |
| days_since_last | 0.018 |
| products | 0.011 |
| tenure_days | 0.000 |

**Which direction** (partial dependence): risk falls from about 51% for one-order customers to about 31% at five orders, then flattens; it stays near 40% for the first three months of silence, then jumps to about 51% between 100 and 130 days.

**SHAP** agrees: mean |SHAP| ranks `orders` first (0.36), then `days_since_last` (0.22) and `spend` (0.20); `tenure_days` is near zero.

**One customer** (the riskiest in the test set): 1 order, £80 spent, 1 product, last seen 273 days ago. Predicted churn 66% against an average of 42%. In CEO language: *"They bought once, a long time ago, for very little. Each of those facts raises the risk."*

## 8. Where it breaks

- **Explanations describe the model, not the world.** SHAP says why the model predicted churn, not why the customer left. Causal claims ("sending vouchers to one-order customers will reduce churn") need experiments.
- **Correlated features share credit unpredictably.** `orders`, `products` and `spend` move together. The model may lean on one, and importance can shift between them from one retrain to the next. Interpret groups of related features together.
- **Built-in importance is biased.** Toward high-cardinality features, and computed on training data. Use it for a quick look only.
- **Partial dependence averages away interactions.** If a feature pushes up for some customers and down for others, the average can look flat. SHAP scatter plots (exercise idea) show the spread.
- **Leakage shows up here first.** If one feature dominates suspiciously (e.g. `recency_days` from 01.9), check it isn't computed from the future. Explanations are a leak detector.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: Shapley values for a model with base 20, A alone 20, B alone 50, both 60. What is odd about A?
2. On paper: permutation importance on the six customers with a different shuffle.
3. Compare permutation importance using accuracy vs ROC-AUC. Does the ranking change?
4. Explain the *least* risky test customer with a waterfall, in two sentences for a retention manager.

**Quiz**

1. What is the difference between a global and a local explanation?
2. Why can built-in tree importance rank a random column above a real feature?
3. Describe permutation importance in one sentence.
4. Compute the Shapley values for the toy model in section 5 again, showing both orders.
5. What do the base value and the SHAP values always add up to?

## 10. Recap

- **Global** questions (what does the model rely on?) and **local** questions (why this customer?) need different tools.
- **Built-in importance** is quick but biased; **permutation importance** measures the drop in a real score on test data.
- **Partial dependence** shows direction and shape: thresholds, plateaus, diminishing returns.
- **SHAP** splits each prediction fairly: base value + one push per feature = the prediction.
- Explanations describe the **model**, not cause and effect, and they are one of the best ways to catch leakage.

**Next:** [05.9 AutoML in industry](../09-automl-pycaret/) (optional), then [05.10 Choosing and saving the final model](../10-choosing-and-saving/).
