# 06.4 · Hierarchical clustering and dendrograms

⏱ about 1 hour · Prerequisites: 06.2, 06.3 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing liked the four personas from 06.3, but two follow-up questions came back:

- *"Which of the four groups are most alike? If we can only afford three campaigns, which two should we merge?"*
- *"Within Champions, are there sub-groups we should know about?"*

K-Means cannot answer either. It gives you exactly $k$ flat groups and says nothing about how they relate. Re-running with $k = 3$ or $k = 8$ produces new groups that need not nest inside the old ones.

What you want is a **family tree** of customers: who is closest to whom, which small groups join into bigger ones, all the way up to "everybody". Then you can cut the tree at whatever level the business needs.

## 2. The idea

Start with every customer standing alone. Find the two most similar and pair them up. Then find the next two most similar (either two lone customers, or a lone customer and an existing pair, or two pairs) and join them. Keep going until everyone is in one big group.

Write down every join and *how far apart* the two groups were when they joined. Drawn as a tree, with the join distance as the height, that record is a **dendrogram**. Low joins = very similar groups. High joins = groups that only meet because you forced everyone together.

Cutting the dendrogram horizontally at some height gives you clusters: every branch the cut crosses becomes one cluster.

## 3. Definition

**Agglomerative (bottom-up) hierarchical clustering:**

1. Start with $n$ clusters of one point each.
2. Merge the two closest clusters.
3. Repeat until one cluster remains ($n - 1$ merges in total).

"Closest" for two *groups* needs a rule, called the **linkage**:

| linkage | distance between groups $P$ and $Q$ | tends to give |
|---|---|---|
| **single** | the *closest* pair, one from each group | long, straggly chains |
| **complete** | the *farthest* pair, one from each group | compact groups of similar width |
| **average** | the average over all pairs | in between |
| **Ward** | how much the total within-cluster squared error would grow if $P$ and $Q$ merged | compact, similar-sized groups; closest to K-Means |

- **Dendrogram**: the tree of merges, with merge distance on the vertical axis.
- **Cutting** the tree at height $h$ (or asking for $k$ clusters) gives a flat clustering.
- **Divisive (top-down)** clustering starts with everyone together and splits; rarely used in practice.

## 4. Formula

With $d(p, q)$ the usual Euclidean distance between two points:

$$d_{\text{single}}(P, Q) = \min_{p \in P,\, q \in Q} d(p, q) \qquad d_{\text{complete}}(P, Q) = \max_{p \in P,\, q \in Q} d(p, q)$$

$$d_{\text{average}}(P, Q) = \frac{1}{|P|\,|Q|}\sum_{p \in P}\sum_{q \in Q} d(p, q)$$

**Ward** merges the pair whose union raises the total within-cluster sum of squares (the inertia from 06.2) the least. The increase for merging $P$ and $Q$ is $\frac{|P|\,|Q|}{|P| + |Q|}\,\lVert c_P - c_Q \rVert^2$, where $c_P$, $c_Q$ are their centroids. (scipy reports Ward heights on a rescaled version of this; the order of merges is what matters.)

## 5. By hand

Five customers, two features as in 06.2 ($x$ = orders, $y$ = items per order):

| | A | B | C | D | E |
|---|---|---|---|---|---|
| $x$ | 1 | 2 | 5 | 6 | 6 |
| $y$ | 1 | 1 | 4 | 5 | 8 |

**Step 0: the distance matrix.**

| | A | B | C | D | E |
|---|---|---|---|---|---|
| A | – | **1.00** | 5.00 | 6.40 | 8.60 |
| B | 1.00 | – | 4.24 | 5.66 | 8.06 |
| C | 5.00 | 4.24 | – | **1.41** | 4.12 |
| D | 6.40 | 5.66 | 1.41 | – | 3.00 |
| E | 8.60 | 8.06 | 4.12 | 3.00 | – |

(For example $d(C, E) = \sqrt{(6-5)^2 + (8-4)^2} = \sqrt{17} = 4.12$.)

**Single linkage, step by step.**

1. Smallest distance: A–B = 1.00. **Merge {A, B} at height 1.00.**
2. Smallest remaining: C–D = 1.41. **Merge {C, D} at 1.41.**
3. Now three groups: {A, B}, {C, D}, {E}. Single-linkage distances (closest pair):
   - {A, B}–{C, D}: min(5.00, 6.40, 4.24, 5.66) = **4.24**
   - {A, B}–E: min(8.60, 8.06) = 8.06
   - {C, D}–E: min(4.12, 3.00) = **3.00**

   **Merge {C, D, E} at 3.00.**
4. Two groups left: {A, B}–{C, D, E}: min of all six cross distances = **4.24**. **Merge everything at 4.24.**

**The same five customers under each linkage** (merge heights):

| merge | single | complete | average | Ward (scipy) |
|---|---|---|---|---|
| {A, B} | 1.00 | 1.00 | 1.00 | 1.00 |
| {C, D} | 1.41 | 1.41 | 1.41 | 1.41 |
| {C, D} + E | 3.00 | 4.12 | 3.56 | 4.08 |
| everyone | 4.24 | 8.60 | 6.33 | 9.69 |

On these five points every linkage merges in the **same order**; only the heights differ. Complete linkage measures to the farthest member, so its last merge is the full width A–E (8.60). Cut any of these trees into two clusters and you get {A, B} and {C, D, E}. The linkages disagree on messier data, and that matters, as section 7 shows.

## 6. In code

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Z = linkage(points, method="single")     # the full merge history
dendrogram(Z, labels=names)              # draw the tree
fcluster(Z, t=2, criterion="maxclust")   # cut into 2 clusters

from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
```

Each row of `Z` is one merge: the two groups merged, the height, and the size of the new group.

## 7. On real data

A random sample of 300 ShopWave customers (a dendrogram of 4,338 leaves is unreadable), RFM logged and standardised, cut into 4 clusters:

| linkage | cluster sizes | silhouette |
|---|---|---|
| Ward | 158, 67, 63, 12 | 0.331 |
| complete | 148, 95, 45, 12 | 0.353 |
| average | 283, 12, 4, 1 | 0.265 |
| **single** | **296, 2, 1, 1** | **0.581** |

Single linkage has by far the **best silhouette** and the **most useless result**: it chains almost everyone into one cluster and leaves three outliers on their own. The silhouette rewards it because three lone outliers are far from everyone. A reminder from 06.3: metrics alone cannot judge a clustering.

Ward's four groups look familiar: 158 customers who bought about twice, 150 days ago; two middle groups of regulars (67 recent, 63 less recent); and **12 big customers** (44 orders, £27,000 on average). Reading the tree from the top: the first split separates the 158 lapsed customers from the 142 active ones, and the very next split peels off the 12 big customers. They are one of the most distinct groups in the data.

Ward and K-Means on the same sample agree only moderately (adjusted Rand index 0.54). Both isolate the 12 big customers (K-Means adds 5 more to make a top group of 17), but they draw the borders between the middle groups differently. Neither is "right".

## 8. Where it breaks

- **Slow and memory-hungry.** The distance matrix has $n^2$ entries. 4,338 customers is fine (19 million distances); 400,000 transactions is not (160 billion).
- **Merges are final.** Once two points are joined they are never separated, even if a later view shows it was a mistake. K-Means can move points between clusters; hierarchical clustering cannot.
- **Single linkage chains.** One bridge of in-between points can glue two separate groups together. Use Ward or complete unless you specifically want chains.
- **Scale matters** exactly as for K-Means: log and standardise first.
- **Dendrograms of thousands of points are unreadable.** Truncate them (`truncate_mode="lastp"`) or cluster a sample.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: run **complete** linkage on the five customers. Confirm the heights 1.00, 1.41, 4.12 and 8.60 from the table.
2. In code: build two round blobs of points joined by a thin line of "bridge" points. Cluster into 2 groups with single and Ward linkage. Which one splits the blobs correctly?
3. On the 300-customer sample, cut the Ward tree into 2, 3, 4 and 5 clusters. Do the clusters **nest** (does every cluster at 5 sit inside one cluster at 4)? Does K-Means with $k$ = 4 and 5 nest?

**Quiz**

1. What does the height of a merge in a dendrogram mean?
2. How do single and complete linkage measure the distance between two groups?
3. Which linkage behaves most like K-Means, and why?
4. Why did single linkage get the best silhouette on real customers while producing a useless result?
5. Give one reason not to use hierarchical clustering on all 400,000 transactions.

## 10. Recap

- Agglomerative clustering merges the two closest groups, $n - 1$ times, and records every merge in a **dendrogram**.
- **Linkage** decides "closest" for groups: single (nearest pair), complete (farthest pair), average, or Ward (smallest increase in inertia).
- On the five customers all linkages merged in the same order: {A, B} at 1.00, {C, D} at 1.41, then E, then everyone; only the heights differed.
- On real customers, single linkage chained 296 of 300 together (yet scored the best silhouette); Ward gave usable groups, split lapsed from active first, then isolated 12 big customers.
- Cut the tree wherever the business needs; the clusters nest, unlike re-running K-Means with a different $k$.

**Next:** [06.5 DBSCAN and Gaussian mixtures](../05-dbscan-gmm/): clusters of any shape, customers labelled as noise, and soft membership.
