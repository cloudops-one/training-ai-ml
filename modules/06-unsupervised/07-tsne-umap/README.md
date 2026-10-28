# 06.7 · t-SNE and UMAP: pictures of high-dimensional data (optional)

⏱ about 30 minutes · Prerequisites: 06.6 (PCA) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The PCA picture of ShopWave's customers (06.6) was useful, but on the handwritten digits it was a mess: most digits overlapped in two dimensions. Marketing has a similar wish for product data with hundreds of columns: *"Show me a map where similar things sit together."*

PCA draws the directions of largest spread. Sometimes what you want instead is a map that keeps **neighbours** together, even if it has to bend and stretch the data to do it.

## 2. The idea

Imagine flattening a crumpled sheet of paper onto a table. PCA photographs the crumpled sheet from the best angle: straight lines stay straight, but parts of the sheet hide behind others. t-SNE instead tries to unfold it, so that points that were touching on the sheet are still touching on the table. To do that it is allowed to stretch some regions and squash others.

The result is a beautiful picture with clear groups. The price is that sizes and distances in the picture can no longer be trusted.

## 3. Definition

- **t-SNE (t-distributed stochastic neighbour embedding)**: a non-linear method that places points in 2-D so that each point's close neighbours in the original space stay close in the picture.
- **Perplexity**: roughly how many neighbours each point tries to keep close (typical values 5–50).
- **UMAP (uniform manifold approximation and projection)**: a newer method with the same aim; faster on large data, keeps somewhat more global layout, and can place new points.
- **Embedding**: the 2-D coordinates the method produces.

Both are for **looking**, not for modelling.

## 4. Formula

t-SNE turns distances into probabilities of being neighbours, once in the original space and once in the picture, and moves the points until the two sets of probabilities match.

In the original space, the chance that $i$ picks $j$ as a neighbour uses a bell curve whose width $\sigma_i$ is set by the perplexity:

$$p_{j|i} = \frac{\exp(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\lVert x_i - x_k\rVert^2 / 2\sigma_i^2)}$$

In the picture, it uses a heavier-tailed curve (the Student-t, which gives the method its "t"):

$$q_{ij} \propto \left(1 + \lVert y_i - y_j\rVert^2\right)^{-1}$$

The points $y$ are moved by gradient descent to minimise the mismatch (the KL divergence) between $p$ and $q$. You do not need to compute this by hand. What matters is the consequence: **only who is near whom is preserved.** Because $\sigma_i$ adapts to local density, dense regions are expanded and sparse ones squashed.

## 5. By hand

There is no short pen-and-paper version of t-SNE, so do this as a reading exercise. Four statements about a t-SNE plot of ShopWave customers. Mark each **trust** or **don't trust**:

1. "Customers A and B sit right next to each other, so they have similar RFM values."
2. "The blue cluster is three times bigger than the red one, so those customers are much more varied."
3. "The green cluster is far from the others, so it is the most different segment."
4. "There are five clear islands, so ShopWave has five types of customer."

Answers: 1 **trust** (local neighbourhoods are what t-SNE keeps). 2 **don't trust** (cluster sizes are rescaled). 3 **don't trust** (gaps between clusters are not meaningful). 4 **be careful**: try other perplexities and seeds; islands that appear in all of them are more believable, and you should check with a real clustering measure (06.3).

## 6. In code

```python
from sklearn.manifold import TSNE
xy = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)
plt.scatter(xy[:, 0], xy[:, 1], c=labels)       # labels only for colouring
```

With `umap-learn` installed (`pip install umap-learn`):

```python
import umap
xy = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X)
```

A common recipe for wide data is PCA to about 30–50 components first, then t-SNE.

## 7. On real data

**Digits (64 dimensions).** On the PCA picture, a point's nearest neighbour has the same label only **57.0%** of the time; on the t-SNE picture **96.9%**, slightly better even than the full 64 pixels (96.4%). t-SNE never saw the labels; it simply kept similar images together.

**Perplexity 5, 30 and 100** give three noticeably different pictures of the same data. None of them is "the" answer.

**The size illusion.** Two groups in 10-D, one with spread 1.0 and one with spread 9.9. In the t-SNE picture their spreads come out as **6.6 and 6.5**: a tenfold difference has vanished.

**ShopWave customers** (exercise 3). Unlike the digits, there are no well-separated islands: loosely connected strands with a smooth gradient from recent to lapsed customers. Customer "segments" are a convenient way to cut a continuum rather than natural groups.

## 8. Where it breaks

- **Cluster sizes and gaps mean little.** Only local neighbourhoods are reliable.
- **Results change with seed and perplexity.** Always compare a few runs.
- **No `transform` for new data** in scikit-learn's t-SNE. You cannot embed tomorrow's customers into today's map.
- **Never use t-SNE output as model features.** The axes have no meaning and the leakage risk is high. Use PCA, or UMAP fitted inside a pipeline, if you need reduced features.
- **Slow on very large data.** Tens of thousands of points is fine; millions need UMAP or sampling.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Run t-SNE with three seeds. What changes and what stays the same?
2. Compare t-SNE on raw digits vs PCA-30 then t-SNE: time and neighbour agreement.
3. Draw ShopWave's scaled RFM table with t-SNE, coloured by recency. Are there natural segments?

**Quiz**

1. What is t-SNE for?
2. Give two differences between PCA and t-SNE.
3. What does perplexity control?
4. In a t-SNE plot, what does a large cluster tell you about its real spread?
5. Why should t-SNE coordinates not be used as model features?

## 10. Recap

- t-SNE and UMAP draw high-dimensional data in 2-D while keeping **neighbours together**.
- Trust **who is near whom**; do not trust cluster sizes or the gaps between clusters.
- Perplexity (t-SNE) and n_neighbors (UMAP) change the picture; compare several settings and seeds.
- They are for exploration and communication, not for modelling.
- On ShopWave's customers they show a continuum, a useful warning before you name "segments".

**Next:** [06.8 Anomaly detection](../08-anomaly-detection/)
