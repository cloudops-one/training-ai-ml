# 02.7 · Decision tree regressor: predicting with questions

⏱ about 1.5 hours · Prerequisites: 02.3, 02.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Customer success wants a rule of thumb, not an equation: *"Tell me, in two or three questions, which customers are worth a phone call."* Something like: *if they ordered in the last week, expect £2,000+; if not for a month, expect £300; if not for three months, write them off.*

A line cannot say that. A line says "every extra day of silence costs £0.70", smoothly, forever. Real behaviour has thresholds: a customer who ordered 5 days ago and one who ordered 6 days ago are the same; 5 days and 95 days are different worlds. You need a model made of **questions and thresholds**, whose answer is a **step**, not a slope.

## 2. The idea

Split the customers into two groups with one yes/no question, so that spend inside each group is as *similar* as possible. Predict the group's average. If a group is still too varied, split it again. Stop when the groups are small or uniform.

```
                 days_since_last ≤ 6.5 ?
                 ┌──────────┴──────────┐
                yes                    no
            mean = £110            mean = £30
          (3 customers)          (3 customers)
```

The result is a flow-chart. Prediction for a new customer: answer the questions, land in a leaf, report the leaf's mean. No formula, no coefficients, and anyone can read it.

## 3. Definition

A **decision tree regressor** predicts a number by routing each row through a sequence of binary questions (*feature ≤ threshold?*) to a **leaf**, and returning the mean target of the training rows that landed in that leaf.

- **Root node**: the first question. **Internal node**: any later question. **Leaf**: a terminal group with a constant prediction.
- **Split**: a (feature, threshold) pair. The tree picks the split that most reduces the **variance** of the target within the groups.
- **Depth**: how many questions deep the tree goes. `max_depth` is its main dial. **`min_samples_leaf`** is the other: the smallest group allowed.
- **Greedy**: the tree picks the best split *now*, one node at a time, never looking ahead.

Lesson 02.7's classification cousin (03.3) uses the same mechanics with a different measure of "similar" (Gini impurity instead of variance).

## 4. Formula

Variance of the target in a group $G$ with $n_G$ rows and mean $\bar{y}_G$:

$$\text{Var}(G) = \frac{1}{n_G}\sum_{i \in G}(y_i - \bar{y}_G)^2$$

(This is the MSE of predicting the group's mean for everyone in it.)

For a candidate split of parent $P$ into left $L$ and right $R$, the **variance reduction** is

$$\Delta = \text{Var}(P) - \left(\frac{n_L}{n_P}\text{Var}(L) + \frac{n_R}{n_P}\text{Var}(R)\right)$$

the parent's variance minus the size-weighted average of the children's. The tree evaluates every feature and every threshold between consecutive sorted values and keeps the split with the largest $\Delta$.

## 5. By hand

Six customers, one feature.

| customer | days_since_last | spend next month |
|---|---|---|
| A | 1 | 120 |
| B | 2 | 100 |
| C | 3 | 110 |
| D | 10 | 40 |
| E | 20 | 30 |
| F | 30 | 20 |

**Parent.** $\bar{y} = 420/6 = 70$. Deviations: 50, 30, 40, −30, −40, −50. Squares: 2500, 900, 1600, 900, 1600, 2500, sum 10,000. $\text{Var}(P) = 10000/6 = \mathbf{1666.7}$.

**Candidate thresholds** sit halfway between consecutive values: 1.5, 2.5, 6.5, 15, 25. Evaluate each:

| threshold | left group | $\bar{y}_L$ | Var(L) | right group | $\bar{y}_R$ | Var(R) | weighted child var | $\Delta$ |
|---|---|---|---|---|---|---|---|---|
| 1.5 | A | 120 | 0 | B C D E F | 60 | 1400 | $\frac{1}{6}0 + \frac{5}{6}1400 = 1166.7$ | 500 |
| 2.5 | A B | 110 | 100 | C D E F | 50 | 1250 | $\frac{2}{6}100 + \frac{4}{6}1250 = 866.7$ | 800 |
| **6.5** | **A B C** | **110** | **66.7** | **D E F** | **30** | **66.7** | $\frac{3}{6}66.7 + \frac{3}{6}66.7 = \mathbf{66.7}$ | **1600** |
| 15 | A B C D | 92.5 | 968.8 | E F | 25 | 25 | $\frac{4}{6}968.8 + \frac{2}{6}25 = 654.2$ | 1012.5 |
| 25 | A B C D E | 80 | 1400 | F | 20 | 0 | $\frac{5}{6}1400 + \frac{1}{6}0 = 1166.7$ | 500 |

Worked example for 6.5, left group {120, 100, 110}: mean 110, deviations 10, −10, 0, squares 100, 100, 0, sum 200, $\text{Var} = 200/3 = 66.7$. Right group {40, 30, 20}: mean 30, same arithmetic, 66.7.

**Winner: `days_since_last ≤ 6.5`**, which removes 1600 of the 1667 units of variance. That is the one-question tree drawn in section 2. A second level would split each side again (e.g. left at 1.5 or 2.5, right at 15 or 25) for tiny further gains; with `max_depth=1` the tree stops here.

**Predicting:** a customer last seen 4 days ago → "≤ 6.5? yes" → **£110**. Last seen 25 days ago → **£30**. Last seen 100 days ago → still **£30**: the tree never extrapolates beyond its leaves, for better and worse.

## 6. In code

```python
from sklearn.tree import DecisionTreeRegressor, export_text
tree = DecisionTreeRegressor(max_depth=1).fit(X, y)
print(export_text(tree, feature_names=["days_since_last"]))
#   |--- days_since_last <= 6.50
#   |   |--- value: [110.00]
#   |--- days_since_last >  6.50
#   |   |--- value: [30.00]
```

The notebook confirms the threshold and leaf values match the table, then draws the tree's step-shaped prediction over the points next to the straight line, grows it to depth 2, 3 and unlimited, and shows the staircase turning into a memorised zig-zag.

## 7. On real data

On the 3,317 real customers with six features (held-out test set as in 02.6), the notebook compares a line with trees of depth 2, 4, 8 and unlimited. Training R² climbs with depth all the way to 1.0 (an unlimited tree memorises every customer); **test** R² peaks around depth 4 and then falls. And on this particular table the tree *loses to the line*: a few wholesalers with enormous spend sit alone in leaves, and the tree's constant-per-leaf predictions cannot follow a target that keeps growing. That is honest and important: trees are not better than lines, they are *differently shaped*, and the data decides.

Then the notebook reads the depth-2 tree as a business rule, which is what customer success asked for.

## 8. Where it breaks

- **Overfitting is the default.** An unlimited tree has one leaf per row. Always set `max_depth` or `min_samples_leaf`, and always judge on held-out data.
- **No extrapolation.** Predictions are leaf means; outside the training range the tree repeats the edge leaf forever. A line at least keeps going.
- **Steps, not slopes.** A truly linear relationship needs many splits to approximate, and still comes out jagged. If the truth is a line, use a line.
- **Unstable.** Move one row and the root split can change, cascading through the whole tree. Random forests (05.2) fix this by averaging hundreds of trees.
- **Greedy.** The best first split may lead to a worse tree than a slightly worse first split would. The tree never reconsiders.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: the left group {A, B, C} from the winning split. Evaluate thresholds 1.5 and 2.5 on it alone. Which second-level split is better, and by how much?
2. In code: on the 20-customer table, fit trees of depth 1–6 on all three numeric features and tabulate training MAE. At what depth does it reach 0? What does that mean?
3. Fit a depth-2 tree and a line on the 20 customers. For a hypothetical customer with 25 visits (beyond the data), what does each predict? Which do you trust more, and why?
4. On the real data, fit `DecisionTreeRegressor(max_depth=3, min_samples_leaf=50)` and print it with `export_text`. Rewrite it as three plain-English rules customer success could use.

**Quiz**

1. What does a leaf predict?
2. How does a regression tree decide which split is best?
3. Why does an unlimited-depth tree get training error 0?
4. A customer's feature value is far outside the training range. What does a tree do? A line?
5. Name one advantage of a tree over a line and one of a line over a tree.

## 10. Recap

- A regression tree asks yes/no questions and predicts the **mean of the leaf**.
- Splits are chosen by **variance reduction**: $\Delta = \text{Var}(P) - $ weighted $\text{Var}$ of children; the six-row example picks `days ≤ 6.5` with $\Delta = 1600$.
- Trees are readable, handle thresholds and interactions naturally, need no scaling, and do not extrapolate.
- Depth controls overfitting; unlimited depth memorises. Judge on held-out data.
- On ShopWave's heavy-tailed spend, a line beats a single tree. Ensembles (Module 05) make trees competitive.

**Next:** [02.P Mini-project](../project/): predict next-quarter spend for every ShopWave customer, and write up which model to ship.
