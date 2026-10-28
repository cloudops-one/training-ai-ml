# 06.6 · PCA: fewer dimensions, same story

⏱ about 2 hours · Prerequisites: 01.8 (variance, correlation), 04.5 (scaling), 06.2 (K-Means) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's merchandising team wants a single chart of "what kinds of customers do we have?". You have three RFM numbers per customer, and you could add dozens more: spend per category, share of orders in December, average basket size. Nobody can look at a 40-column scatter plot.

The same problem hurts models. With many columns, distance-based methods like KNN and K-Means start to fail, models train slowly, and correlated columns (orders, spend, products) repeat the same information several times.

You need a way to say: *keep the few directions that carry most of the information, drop the rest.*

## 2. The idea

Picture a cloud of points shaped like a cigar, tilted at an angle. You could describe each point with its x and y. But most of the cloud's spread runs along the cigar. If you rotate your axes so the first axis runs along the cigar, the first coordinate tells you almost everything, and the second is mostly small wobble.

**PCA (principal component analysis)** finds that rotation. The first new axis points along the direction of largest spread; the second is the largest spread at right angles to the first; and so on. Then you keep only the first few axes.

It is like photographing a 3-D object from the angle that shows the most: one picture instead of a 3-D model, and you lose surprisingly little.

**Why this matters (the curse of dimensionality).** Scatter 500 random points in a cube and measure one point's distance to the others. In 2-D, the farthest point is 67 times as far as the nearest. In 1,000-D it is only 1.1 times as far: every point is about the same distance from every other, so "nearest neighbour" means almost nothing. Fewer, better dimensions bring meaning back.

## 3. Definition

- **Principal component (PC)**: a direction in feature space (a list of weights, one per original column). PC1 has the largest variance of the data projected onto it; each later PC has the largest remaining variance while being at right angles to all earlier ones.
- **Score**: a point's coordinate along a PC, i.e. its weighted sum of centred features.
- **Explained variance ratio**: the share of total variance captured by each PC. They sum to 1.
- **Scree plot**: a bar chart of explained variance ratios, used to decide how many PCs to keep.
- **Covariance matrix**: a square table whose diagonal holds each column's variance and whose off-diagonal entries hold the covariance of each pair of columns.

PCA is **unsupervised**: it never looks at the target.

## 4. Formula

With centred data $X_c$ (each column minus its mean) and $n$ rows:

$$C = \frac{1}{n-1} X_c^\top X_c$$

The principal components are the **eigenvectors** $v$ of $C$, and the variance along each is its **eigenvalue** $\lambda$:

$$C\,v = \lambda\,v$$

An eigenvector is a direction that $C$ only stretches, never turns; the stretch factor is the eigenvalue. For a 2×2 matrix the eigenvalues solve

$$\lambda^2 - (\text{trace})\,\lambda + \det = 0$$

Explained variance ratio of PC $k$: $\lambda_k / \sum_j \lambda_j$. Score of a point: $z = (x - \bar{x}) \cdot v$.

## 5. By hand

Five customers with two scaled features:

| customer | $x_1$ | $x_2$ |
|---|---|---|
| A | 2 | 1 |
| B | 3 | 3 |
| C | 4 | 3 |
| D | 5 | 5 |
| E | 6 | 8 |

**Step 1: centre.** Means are 4 and 4. Centred points: (−2, −3), (−1, −1), (0, −1), (1, 1), (2, 4).

**Step 2: covariance matrix** (divide by $n-1 = 4$).

- $\text{Var}(x_1) = (4 + 1 + 0 + 1 + 4)/4 = 10/4 = 2.5$
- $\text{Var}(x_2) = (9 + 1 + 1 + 1 + 16)/4 = 28/4 = 7$
- $\text{Cov}(x_1, x_2) = (6 + 1 + 0 + 1 + 8)/4 = 16/4 = 4$

$$C = \begin{pmatrix} 2.5 & 4 \\ 4 & 7 \end{pmatrix}$$

**Step 3: eigenvalues.** Trace = 2.5 + 7 = 9.5. Determinant = 2.5 × 7 − 4 × 4 = 1.5.

$$\lambda = \frac{9.5 \pm \sqrt{9.5^2 - 4 \times 1.5}}{2} = \frac{9.5 \pm \sqrt{84.25}}{2} = \frac{9.5 \pm 9.179}{2}$$

So $\lambda_1 = 9.339$ and $\lambda_2 = 0.161$.

**Step 4: PC1 direction.** From $(2.5 - 9.339)\,v_1 + 4\,v_2 = 0$: $v_2 = 1.710\,v_1$. Normalise to length 1: $v = (1, 1.710)/\sqrt{1 + 1.710^2} = (0.505, 0.863)$.

**Step 5: explained variance.** $9.339 / (9.339 + 0.161) = 9.339 / 9.5 = 0.983$. **PC1 alone keeps 98.3% of the variance.**

**Step 6: scores.** Customer A: $(-2)(0.505) + (-3)(0.863) = -1.01 - 2.59 = -3.60$. All five: −3.60, −1.37, −0.86, +1.37, +4.46. Two numbers per customer became one, and almost nothing was lost.

Check: the variance of those five scores is 9.339, exactly $\lambda_1$.

## 6. In code

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)          # X already scaled
pca.components_                            # one row per PC: the weights
pca.explained_variance_ratio_              # share of variance per PC
scores = pca.transform(X)                  # new coordinates
```

The notebook does the five-customer example three ways (by formula, with `np.linalg.eigh`, with `PCA`) and gets 0.505, 0.863 and 98.3% each time. scikit-learn may flip a component's sign; the direction is the same line either way.

Inside a model, put PCA in the pipeline so it is fitted only on training folds:

```python
make_pipeline(StandardScaler(), PCA(n_components=20), LogisticRegression())
```

## 7. On real data

**Handwritten digits** (8×8 images, 64 pixel columns). PC1 explains 14.9% and PC2 13.6%, so two components keep about 29%; that is already enough for 0s, 4s and 6s to form their own regions on a 2-D plot. You need 13 components for 80% of the variance, 21 for 90% and 29 for 95%.

As a preprocessing step for logistic regression (5-fold CV accuracy):

| components | 2 | 5 | 10 | 20 | 40 | 64 |
|---|---|---|---|---|---|---|
| accuracy | 0.535 | 0.771 | 0.840 | 0.899 | 0.914 | 0.920 |

Twenty components get within about 2 points of all 64.

**ShopWave RFM.** On the raw table, PC1 is simply monetary (weights 0, 0, 1) and explains essentially 100% of the variance: pounds vary in the thousands, so they swamp days and order counts. After log + scaling, the ratios are 0.752, 0.187 and 0.061, and PC1's weights are −0.51 (recency), +0.62 (frequency), +0.60 (monetary). That is a **customer value** axis: recent, frequent, high-spending customers score high. Colouring a 2-D PCA plot by four K-Means segments shows them lined up mostly along that axis.

## 8. Where it breaks

- **Scale first.** PCA maximises variance, and variance depends on units. Unscaled, the column with the biggest numbers wins (pounds beat orders), whatever its importance.
- **Variance is not usefulness.** PCA never sees the target. If the signal lives in a low-variance column, PCA can throw it away (exercise 4 shows a model dropping from 99.6% to a coin flip).
- **Components are hard to explain.** "PC3 went up" means nothing to a stakeholder. If you need explanations, keep original features or name the components from their weights.
- **Straight lines only.** PCA finds flat directions. Curved structure (a spiral, a ring) needs other tools, such as t-SNE and UMAP for pictures (06.7).
- **Outliers pull components.** One huge customer can tilt PC1 towards themselves. Log-transform or cap first.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: PCA of (1, 2), (2, 4), (3, 6), (4, 8). What is special about this data?
2. Reconstruct digits from 5, 15 and 40 components with `inverse_transform`. When can you read them?
3. Show the first four components of the digits as 8×8 images and describe them.
4. Build a dataset where PCA discards the signal, and measure the damage.

**Quiz**

1. In plain words, what is the curse of dimensionality?
2. What is the first principal component?
3. Why must you scale before PCA?
4. Using the by-hand example, what share of variance does PC1 explain, and how is it computed?
5. Name two situations where you should *not* use PCA.

## 10. Recap

- High-dimensional data makes distances meaningless and models slow; PCA keeps the few directions with the most spread.
- PCs are the eigenvectors of the covariance matrix; their eigenvalues are the variance each one keeps.
- Explained variance ratio and the scree plot tell you how many components to keep.
- Scale (and usually log) first, and put PCA inside the pipeline.
- PCA keeps spread, not predictive power, and its components are hard to explain.

**Next:** [06.7 t-SNE and UMAP](../07-tsne-umap/) (optional), or go straight to [06.8 Anomaly detection](../08-anomaly-detection/).
