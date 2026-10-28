# 03.3 · Decision tree classifier: Gini impurity by hand

⏱ about 2 hours · Prerequisites: 03.1, 02.7 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention liked the KNN voucher list but asked a fair question: *"Why these customers? What is the rule?"* KNN cannot answer; it only says "people like them left".

A decision tree can. It learns rules of the form *"if visits ≤ 9 and days since last order > 43, then churn"*, which you can print on one page and hand to the retention team. In lesson 00.1 you watched a tree find rules for fraud. Now you learn how it chooses them, and you grow one yourself with pencil and paper.

## 2. The idea

Play "twenty questions" with the customer table. You want to ask the question that **best separates churners from stayers**. "Did they visit more than 9 times?" splits the 20 customers into a group that is mostly stayers and a group that is mostly churners. That is a good question. "Are they in the UK?" splits them into two groups that are both about half and half. That is a useless question.

A tree is built by asking the best question, then, inside each resulting group, asking the next best question, and so on until the groups are pure enough or small enough. To rank questions, you need a number that says *how mixed* a group is. That number is **Gini impurity**.

## 3. Definition

**Decision tree classifier**: a flow-chart of yes/no questions about features, learned from data, where each leaf predicts the majority class of the training rows that reached it.

- **Node**: a question. **Root**: the first one. **Leaf**: a final prediction.
- **Split**: a feature and a threshold, e.g. `visits ≤ 9`.
- **Gini impurity**: how mixed the classes are in a group; 0 means pure, 0.5 is the worst for two classes.
- **Gain**: how much a split reduces impurity. The tree picks the split with the largest gain.
- **Depth**: the longest chain of questions. The main lever for controlling overfitting.

## 4. Formula

For a group where a fraction $p$ belongs to class 1 (and $1 - p$ to class 0):

$$G = 1 - p^2 - (1-p)^2$$

In words: pick two rows at random from the group; $G$ is the probability they have *different* classes. All one class → $G = 0$. Fifty-fifty → $G = 0.5$.

For a split that sends $n_L$ rows left and $n_R$ right, the **weighted impurity after the split** is

$$G_{\text{split}} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R$$

and the **gain** is $G_{\text{parent}} - G_{\text{split}}$. The tree tries every feature and every threshold between consecutive values, computes $G_{\text{split}}$ for each, and keeps the smallest.

## 5. By hand

The 20 customers: 9 churned, 11 stayed.

**Step 1: impurity of the whole group (the root).**

$p = 9/20 = 0.45$

$G_{\text{root}} = 1 - 0.45^2 - 0.55^2 = 1 - 0.2025 - 0.3025 = \mathbf{0.495}$

Almost maximally mixed, as expected for a nearly even split.

**Step 2: candidate split A, `visits ≤ 9`.**

| side | customers | n | churned | $p$ | $G$ |
|---|---|---|---|---|---|
| left (≤ 9) | C001 C004 C005 C007 C009 C010 C011 C017 C018 C020 | 10 | 8 | 0.8 | $1 - 0.64 - 0.04 = 0.32$ |
| right (> 9) | C002 C003 C006 C008 C012 C013 C014 C015 C016 C019 | 10 | 1 | 0.1 | $1 - 0.01 - 0.81 = 0.18$ |

$G_{\text{split}} = \frac{10}{20}(0.32) + \frac{10}{20}(0.18) = 0.16 + 0.09 = \mathbf{0.25}$

Gain $= 0.495 - 0.25 = 0.245$.

**Step 3: candidate split B, `days_since_last ≤ 45.5`.**

| side | n | churned | $p$ | $G$ |
|---|---|---|---|---|
| left (≤ 45.5) | 16 | 5 | 0.3125 | $1 - 0.0977 - 0.4727 = 0.430$ |
| right (> 45.5) | 4 | 4 | 1.0 | $0$ |

$G_{\text{split}} = \frac{16}{20}(0.430) + \frac{4}{20}(0) = \mathbf{0.344}$. Gain $= 0.151$.

Split B makes one perfectly pure leaf (everyone who waited more than 45 days churned) but leaves 16 customers still mixed. Split A is better overall: **the root question is `visits ≤ 9`**. (The notebook checks every possible threshold on both features; 0.25 is the minimum.)

**Step 4: grow the left branch.** Ten customers, 8 churned, $G = 0.32$. Best next split: `visits ≤ 5` → left: C001 C007 C009 C010 C018, 5 of 5 churned, $G = 0$; right: C004 C005 C011 C017 C020, 3 of 5 churned, $G = 0.48$. Weighted: $\frac{5}{10}(0) + \frac{5}{10}(0.48) = 0.24$.

**Step 5: grow the right branch.** Ten customers, 1 churned (C019: 12 visits, 34 days), $G = 0.18$. Best split: `days_since_last ≤ 31.5` → left: C003 C006 C008 C012 C013 C014 C016, 7 customers, 0 churned, $G = 0$; right: C002 C015 C019, 3 customers, 1 churned, $G = 1 - (1/3)^2 - (2/3)^2 = 0.444$. Weighted: $\frac{7}{10}(0) + \frac{3}{10}(0.444) = 0.133$.

Both right-hand leaves still predict "stay", so this split changes no predictions; it only sharpens the probabilities (0% vs 33% churn).

**The depth-2 tree you grew:**

```
visits ≤ 9?
├── yes: visits ≤ 5?
│   ├── yes → churn   (5/5 churned)
│   └── no  → churn   (3/5 churned)
└── no:  days_since_last ≤ 31.5?
    ├── yes → stay    (0/7 churned)
    └── no  → stay    (1/3 churned)
```

Accuracy on the 20 customers: 17/20 = **85%**, the same as the single hand rule from 03.1, because depth 2 did not change any predictions, only confidence. That is a lesson in itself: more questions are not always more accuracy.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will write `gini()` in three lines, reproduce the split table for every threshold, then let scikit-learn do it:

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X20, y20)
print(export_text(tree, feature_names=list(X20.columns)))
plot_tree(tree, feature_names=list(X20.columns), class_names=["stay", "churn"], filled=True)
```

Each box in the plot shows the split, the Gini, the row count and the class counts. Compare them with your table: they match.

## 7. On real data

On the real churn table (five features), the notebook grows trees of depth 1, 2, 3, 5 and unlimited and scores each on training and on test customers:

| depth | train accuracy | test accuracy |
|---|---|---|
| 1 | 66% | 66% |
| 2 | 66% | 64% |
| 3 | 68% | 66% |
| 5 | 71% | 65% |
| unlimited | **100%** | **60%** |

The unlimited tree memorises every training customer and does *worse* than a one-question stump on new customers. That gap between train and test is **overfitting**, and `max_depth` is the first knob for controlling it. Module 04 makes this rigorous.

The depth-2 tree's rules are readable business logic: *fewer than 3 orders and no order for 109+ days → churn*.

## 8. Where it breaks

- **Overfitting by default.** An unlimited tree fits the training data perfectly and generalises badly. Always limit depth or minimum leaf size.
- **Unstable.** Change a few training rows and the root split can change, which changes the whole tree. Random forests (Module 05) fix this by averaging many trees.
- **Axis-aligned.** Each split is one feature at a time, so a diagonal boundary needs a staircase of many splits.
- **Greedy.** The best first question is not always part of the best tree. The tree never goes back.
- **Biased towards features with many values.** A feature with 1,000 distinct values offers 999 thresholds to try and wins splits by chance.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: compute Gini impurity for groups with class counts (10, 0), (5, 5), (7, 3) and (3, 7). What do you notice about the last two?
2. By hand, then check: on the 20 customers, compute $G_{\text{split}}$ for `items_last_month ≤ 4.5`. Is it better or worse than `visits ≤ 9`?
3. In code: use **entropy** instead of Gini (`criterion="entropy"`). Entropy is $-p \log_2 p - (1-p)\log_2(1-p)$. Does the depth-2 tree change?
4. On the real table, plot test accuracy against `max_depth` from 1 to 20. Where does it peak? Then try `min_samples_leaf=50` with unlimited depth.

**Quiz**

1. What does Gini impurity 0 mean? And 0.5?
2. A group of 8 rows has 6 churned. Compute its Gini.
3. How does a tree choose the split at each node?
4. Why did the depth-2 tree score the same 85% as the one-rule classifier from 03.1?
5. Train accuracy 100%, test accuracy 60%. What happened and which parameter fixes it?

## 10. Recap

- A decision tree asks the question that most reduces **Gini impurity**, $G = 1 - p^2 - (1-p)^2$, then repeats inside each branch.
- On 20 customers, `visits ≤ 9` cuts impurity from 0.495 to 0.25; no other single question does better.
- Trees are **readable**: you can print the rules and hand them to the business.
- Unlimited trees memorise; `max_depth` (or `min_samples_leaf`) is the first defence against overfitting.
- Trees need **no scaling** (they only compare a feature with a threshold), unlike KNN.

**Next:** [03.4 Logistic regression](../04-logistic-regression/): the model that gives a proper probability, and the one you will use most in practice.
