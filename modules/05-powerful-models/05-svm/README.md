# 05.5 · Support vector machines *(optional)*

⏱ about 1.5 hours · Prerequisites: 03.2 (KNN, distances, scaling), 03.4 (linear decision boundary), 04.10 (regularisation) · Notebook: [lesson.ipynb](lesson.ipynb)

> **Optional lesson.** Support vector machines were the state of the art for classification from the late 1990s to about 2010 and still appear in interviews and older systems. If you are short on time, read sections 1–3 and 10 and move on to 05.6.

---

## 1. The problem

Logistic regression (03.4) draws *a* line between churners and stayers: the line that makes the predicted probabilities fit the data best. But look at a scatter of two well-separated groups and you could draw dozens of lines that separate them perfectly. Which is the *best* one?

Intuitively, the one that stays **as far as possible from both groups**. A line that grazes the edge of one group will misclassify the next customer who lands slightly to that side. A line down the middle of the gap leaves room for error on both sides. That is the support vector machine's idea, and it led to one of the most elegant algorithms in machine learning.

## 2. The idea

Imagine building a road between two villages, and you want the **widest possible road** that keeps every house off the tarmac. The centre line of that road is the decision boundary. The road's edges are the **margins**. The handful of houses that touch the road's edges are the only ones that matter: move any other house and the road stays put; move one of the edge houses and the road must shift. Those edge houses are the **support vectors**.

```
     ●   ●                          ●  = churned
  ●     ●  ●  ⊙ ‥‥‥‥‥‥‥‥‥‥‥‥‥‥‥      ○  = stayed
     ●       ‥                 ‥    ⊙ ◎ = support vectors (touch the margin)
            ‥  ─ ─ boundary ─ ─ ‥
           ‥                 ‥   ○
          ‥‥‥‥‥‥‥‥‥‥‥‥‥‥‥‥ ◎    ○   ○
                            ○   ○
              ◂── margin ──▸
```

When the groups overlap (they always do with real churn), some houses *must* sit on the road. The parameter **C** sets how much each such violation costs.

And when no straight road works at all, the **kernel trick** lets the SVM build a curved one.

## 3. Definition

- **Support vector machine (SVM)**: a classifier that finds the boundary with the **largest margin** between classes, allowing some violations.
- **Margin**: the width of the empty corridor around the boundary.
- **Support vectors**: the training points on or inside the margin. Only they determine the boundary.
- **Hinge loss** (in words): no penalty for a point that is on the correct side *and* outside the margin; a penalty that grows linearly with how far a point intrudes into the margin or onto the wrong side.
- **C**: the cost of each margin violation. Small `C` → wide, tolerant margin (more regularisation). Large `C` → narrow margin that tries to classify every training point correctly.
- **Kernel**: a function measuring similarity between two points. With the **RBF** (radial basis function) kernel, similarity decays with distance, which lets the boundary curve.
- **gamma** (RBF): how quickly similarity decays. Small = far-reaching influence, smooth boundary. Large = very local influence, wiggly boundary.
- **SVR** (support vector regression): the same idea for numbers. Fit a tube of half-width $\varepsilon$ around the data and ignore errors inside it.

## 4. Formula

A linear SVM's boundary is the set of points where

$$w \cdot x + b = 0$$

(the same form as logistic regression's $z = 0$). The margin edges are where $w \cdot x + b = \pm 1$, and the distance between them is

$$\text{margin width} = \frac{2}{\lVert w \rVert}, \qquad \lVert w \rVert = \sqrt{w_1^2 + w_2^2 + \dots}$$

So **widening the margin means making $\lVert w \rVert$ small**. The SVM minimises

$$\tfrac{1}{2}\lVert w \rVert^2 \;+\; C \sum_i \max\bigl(0,\; 1 - y_i\,(w \cdot x_i + b)\bigr)$$

where $y_i = \pm 1$. The first term asks for a wide margin; the second (the **hinge loss**) charges $C$ for each unit of intrusion. You have seen this shape before: it is regularisation (04.10) with a different loss.

The RBF kernel's similarity between points $x$ and $x'$:

$$K(x, x') = e^{-\gamma \lVert x - x' \rVert^2}$$

Identical points → 1. Far-apart points → close to 0. Large $\gamma$ → similarity falls off fast.

## 5. By hand

**Margin width.** The notebook's linear SVM on 40 points has $w = (-0.254, -0.838)$:

$$\lVert w \rVert = \sqrt{0.254^2 + 0.838^2} = \sqrt{0.0645 + 0.7022} = \sqrt{0.7667} = 0.876$$
$$\text{width} = 2 / 0.876 = 2.28$$

**Effect of C** on the same 40 points:

| C | margin width | support vectors |
|---|---|---|
| 0.01 | 5.71 | 18 of 40 |
| 1 | 2.28 | 3 of 40 |
| 100 | 2.28 | 3 of 40 |

Cheap mistakes ($C = 0.01$) → a corridor two and a half times wider, with 18 points allowed inside it. Expensive mistakes → the corridor shrinks until just 3 points hold it up. These two groups are cleanly separable, so beyond $C = 1$ nothing changes.

**RBF similarity.** Two customers at distance 1 (after scaling): with $\gamma = 0.1$, $K = e^{-0.1} = 0.90$, "very similar". With $\gamma = 10$, $K = e^{-10} = 0.00005$, "unrelated". That is why large gamma makes each training point an island and the boundary wraps around individual points.

## 6. In code

```python
from sklearn.svm import SVC, SVR

linear = SVC(kernel="linear", C=1).fit(X, y)
linear.coef_, linear.support_vectors_      # w, and the points that hold the boundary up

curved = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale", probability=True))
curved.fit(X_train, y_train)               # always scale first

svr = SVR(kernel="linear", C=100, epsilon=5)   # regression version
```

`probability=True` is needed for `predict_proba` (and AUC). It runs an extra internal cross-validation, so training is slower.

## 7. On real data

**Two moons** (300 points, no straight line separates them), 5-fold CV accuracy:

| kernel | gamma | CV accuracy | support vectors |
|---|---|---|---|
| linear | – | 0.837 | 110 |
| RBF | 0.1 | 0.837 | 127 |
| RBF | 1 | 0.957 | 73 |
| RBF | 10 | 0.960 | 110 |
| RBF | 100 | 0.927 | 257 |

The RBF kernel bends the boundary around the moons. At gamma 100 training accuracy rises to 0.983 but CV accuracy drops, and almost every point becomes a support vector: overfitting.

**ShopWave churn**: an RBF SVM without scaling scores CV AUC 0.718; with scaling 0.731 (test 0.719). Tuning `C` reaches 0.734. Respectable, but below logistic regression (0.740), and slower to train.

**Scaling with data size**: on a synthetic 10-feature table, the SVM trains faster than a 100-tree random forest at 2,000 rows, but each doubling of the rows multiplies its training time by 3–4 (about 0.01 s → 0.03 → 0.10 → 0.36 s from 2,000 to 16,000 rows on one laptop), while the forest's only about doubles (0.08 → 0.59 s). Projecting those growth rates, the SVM falls behind somewhere in the tens of thousands of rows; at a million rows (six more doublings) it would need roughly 0.36 × 3.7⁶ ≈ 900 s, about a quarter of an hour, against well under a minute for the forest. And that is a single fit, before any tuning. Your exact times will differ; the growth rates will not. (For a clean timing on your own machine, close other heavy programs first.)

## 8. Where it breaks

- **Slow on big data.** Training cost grows roughly with the square of the number of rows. Fine for thousands, painful for millions.
- **Must scale.** Distances and margins are meaningless when one feature is in thousands of pounds and another in single orders.
- **No native probabilities.** `probability=True` bolts them on with an extra cross-validation (slower, and sometimes poorly calibrated).
- **No native missing values or categories**: impute and encode first.
- **Two interacting knobs** (`C`, `gamma`) that must be tuned together.
- **Hard to explain**: a curved boundary defined by dozens of support vectors in five dimensions does not translate into a sentence for the CEO.
- **Where it still shines**: small datasets with many features, such as text classification with thousands of word columns, where a linear SVM is fast and strong.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: a linear SVM has $w = (3, 4)$. What is the margin width?
2. Remove one non-support point from the blobs and refit; then remove one support vector and refit. Which changes the boundary?
3. Grid of `C` × `gamma` on the moons: which combination is best, and which overfits?
4. Tune `C` for a scaled RBF SVM on churn. Can it beat 0.740 CV AUC?

**Quiz**

1. What are support vectors?
2. What does a small `C` do to the margin, compared with a large `C`?
3. What does the RBF kernel let an SVM do, and what does `gamma` control?
4. Why must you scale features before an SVM?
5. Give two reasons SVMs are used less than tree ensembles on business tables today.

## 10. Recap

- An SVM finds the boundary with the **widest margin**: width $= 2 / \lVert w \rVert$.
- Only the **support vectors**, the points on or inside the margin, determine the boundary.
- **C** prices margin violations: small C = wide, tolerant margin; large C = narrow, strict margin.
- The **RBF kernel** curves the boundary; **gamma** sets how local each point's influence is. High gamma overfits.
- Always **scale**. On ShopWave churn a tuned SVM reaches CV AUC 0.734, below logistic regression, and it scales poorly to large data, which is why tree ensembles took over tabular problems.

**Next:** [05.6 Hyperparameter tuning](../06-hyperparameter-tuning/): stop guessing `C`, `gamma`, learning rates and tree sizes, and search for them properly.
