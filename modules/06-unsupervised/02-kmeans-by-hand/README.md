# 06.2 · K-Means by hand

⏱ about 2 hours · Prerequisites: 06.1, 03.2 (distance), 04.5 (scaling) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing wants customer groups. In 06.1 a single line, `KMeans(n_clusters=4).fit_predict(X)`, produced four groups, and one of them brought in two thirds of all revenue. Before you put that in front of the marketing director, you need to be able to answer the obvious questions:

- *How* did it decide who goes in which group?
- Would it give the same groups if you ran it again?
- Why did you log and scale the numbers first? What happens if you do not?

The only way to answer with confidence is to run the algorithm yourself, by hand, on a handful of customers. K-Means is simple enough that you can.

## 2. The idea

Picture eight customers as pins on a map, and two flags you can move. The rules of the game:

1. **Place** the two flags somewhere (anywhere to begin with).
2. **Assign**: every pin joins the team of its nearest flag.
3. **Move**: each flag walks to the middle of its team.
4. **Repeat** 2 and 3 until no pin changes team.

That is the whole algorithm. Each flag is a **centroid**; each team is a **cluster**. Moving the flag to the middle of its team can only bring it closer to its members on average, and re-assigning pins to the nearest flag can only shorten their distances. So the total spread keeps falling until it cannot fall any more, and the flags stop.

## 3. Definition

**K-Means** splits $n$ points into $k$ clusters by repeating two steps until assignments stop changing:

- **Assignment step**: put each point in the cluster whose centroid is nearest (Euclidean distance).
- **Update step**: move each centroid to the **mean** of the points assigned to it.

Vocabulary:

- **$k$**: the number of clusters. *You* choose it (06.3 shows how).
- **Centroid**: the mean position of a cluster's points. It does not have to be one of the customers.
- **Inertia** (within-cluster sum of squares): the total squared distance from each point to its own centroid. Small inertia = tight clusters. K-Means is trying to make it as small as it can.
- **Convergence**: the moment assignments stop changing.
- **Initialisation**: where the centroids start. Bad starts can lead to bad final answers; **k-means++** chooses spread-out starting points to avoid this.

## 4. Formula

**Distance** from point $p$ to centroid $c$ (two features):

$$d(p, c) = \sqrt{(p_1 - c_1)^2 + (p_2 - c_2)^2}$$

**Centroid** of a cluster $S$ with $|S|$ points:

$$c = \left(\frac{1}{|S|}\sum_{p \in S} p_1,\ \ \frac{1}{|S|}\sum_{p \in S} p_2\right)$$

**Inertia** with $k$ clusters:

$$\text{inertia} = \sum_{j=1}^{k} \sum_{p \in S_j} d(p, c_j)^2$$

Squared distances, summed over every point to its own centroid. Each assignment step and each update step can only lower this number (or leave it the same), which is why K-Means always stops.

## 5. By hand

Eight ShopWave customers, two features: $x$ = orders last year, $y$ = average items per order.

| customer | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| orders ($x$) | 1 | 1 | 2 | 4 | 6 | 7 | 7 | 9 |
| items ($y$) | 1 | 3 | 2 | 3 | 6 | 5 | 8 | 7 |

We want $k = 2$ groups. Start the two centroids on customers **A (1, 1)** and **D (4, 3)**. A deliberately lopsided start, so you can watch the algorithm fix it.

### Iteration 1

**Assign.** Distance from every customer to each centroid. Example for E: to $c_1$: $\sqrt{(6-1)^2 + (6-1)^2} = \sqrt{50} = 7.07$; to $c_2$: $\sqrt{(6-4)^2 + (6-3)^2} = \sqrt{13} = 3.61$. Nearer to $c_2$.

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| to $c_1$ (1, 1) | **0.00** | **2.00** | **1.41** | 3.61 | 7.07 | 7.21 | 9.22 | 10.00 |
| to $c_2$ (4, 3) | 3.61 | 3.00 | 2.24 | **0.00** | **3.61** | **3.61** | **5.83** | **6.40** |
| cluster | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 |

**Update.** New centroids are the means of each team:

- $c_1$ = mean of A, B, C = $\left(\frac{1+1+2}{3}, \frac{1+3+2}{3}\right) = (1.33,\ 2.00)$
- $c_2$ = mean of D, E, F, G, H = $\left(\frac{4+6+7+7+9}{5}, \frac{3+6+5+8+7}{5}\right) = (6.60,\ 5.80)$

### Iteration 2

**Assign** with the new centroids. Check D yourself: to $c_1$: $\sqrt{(4-1.33)^2 + (3-2)^2} = \sqrt{8.11} = 2.85$; to $c_2$: $\sqrt{(4-6.6)^2 + (3-5.8)^2} = \sqrt{14.6} = 3.82$.

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| to $c_1$ (1.33, 2.00) | **1.05** | **1.05** | **0.67** | **2.85** | 6.15 | 6.41 | 8.25 | 9.15 |
| to $c_2$ (6.60, 5.80) | 7.38 | 6.26 | 5.97 | 3.82 | **0.63** | **0.89** | **2.24** | **2.68** |
| cluster | 1 | 1 | 1 | **1** | 2 | 2 | 2 | 2 |

**D has switched teams.** Once $c_2$ moved towards the big buyers, D was closer to the small ones.

**Update:**

- $c_1$ = mean of A, B, C, D = $\left(\frac{1+1+2+4}{4}, \frac{1+3+2+3}{4}\right) = (2.00,\ 2.25)$
- $c_2$ = mean of E, F, G, H = $\left(\frac{6+7+7+9}{4}, \frac{6+5+8+7}{4}\right) = (7.25,\ 6.50)$

### Iteration 3

Assign again with $(2.00, 2.25)$ and $(7.25, 6.50)$: every customer stays where it was. **Converged.** Two clusters: {A, B, C, D} (few orders, small baskets) and {E, F, G, H} (many orders, big baskets).

### Inertia

Squared distance from each customer to its own final centroid:

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| $(x - c_x)^2$ | 1 | 1 | 0 | 4 | 1.5625 | 0.0625 | 0.0625 | 3.0625 |
| $(y - c_y)^2$ | 1.5625 | 0.5625 | 0.0625 | 0.5625 | 0.25 | 2.25 | 2.25 | 0.25 |
| sum | 2.5625 | 1.5625 | 0.0625 | 4.5625 | 1.8125 | 2.3125 | 2.3125 | 3.3125 |

Cluster 1: 8.75. Cluster 2: 9.75. **Inertia = 18.5.**

Measured at each step (each customer to the centroid it was assigned to), inertia fell **107 → 24.18 → 18.5**. That steady fall is why K-Means is guaranteed to stop.

## 6. In code

The notebook first writes the two steps in NumPy, about ten lines, and prints the same tables you made. Then scikit-learn, started from the same two customers:

```python
from sklearn.cluster import KMeans

start = points[[0, 3]]                              # customers A and D
km = KMeans(n_clusters=2, init=start, n_init=1).fit(points)
km.cluster_centers_     # [[2.00, 2.25], [7.25, 6.50]]
km.inertia_             # 18.5
km.labels_              # [0 0 0 0 1 1 1 1]
```

Normally you let scikit-learn choose the start: `init="k-means++"` (the default) and `n_init=10`, meaning "try 10 different starts and keep the one with the lowest inertia".

## 7. On real data

Two lessons from the notebook.

**Starting points matter.** With $k = 3$ on the same eight customers, ten random starts finish at three different answers: {A, B, C, D}, {E, F}, {G, H} with inertia **12.25**; D on its own with **12.42**; or the big buyers split diagonally into {E, G} and {F, H} with **15.25**. K-Means only guarantees that it stops, not that it finds the best split; it can get stuck in a *local minimum*. k-means++ starts plus several restarts (`n_init=10`) find 12.25 reliably.

**Scale decides everything.** On all 4,338 ShopWave customers:

| input | cluster sizes ($k = 4$) |
|---|---|
| raw recency (days), frequency (orders), monetary (£) | **4,300**, 31, 5, 2 |
| log, then standardised | 1,635, 1,178, 826, 699 |

On raw numbers, monetary runs into the hundreds of thousands, so it swamps the other two: K-Means puts almost everyone in one group and spends the other three groups on a handful of giant spenders. After log and scaling, all three features count, and the four groups are of a size marketing can actually use.

A tiny example of why: customer P has 2 orders and £500 spend, Q has 20 orders and £520, R has 2 orders and £900. In raw units, $d(P, Q) = \sqrt{18^2 + 20^2} = 26.9$ and $d(P, R) = \sqrt{0^2 + 400^2} = 400$. K-Means would call P and Q near-identical, even though Q orders ten times as often.

## 8. Where it breaks

- **You must choose $k$.** K-Means will happily give you 2 or 20 clusters. 06.3 covers how to choose.
- **Round clusters only.** Because every point joins its nearest centroid, clusters are carved into convex, roughly ball-shaped regions. Curved or stretched groups get chopped up; DBSCAN (06.5) handles them.
- **Sensitive to scale and outliers.** Unscaled features dominate (above), and a single extreme customer can drag a centroid or claim a cluster of its own. Log-transform skewed features first.
- **Local minima.** Always use several starts (`n_init`); compare results across seeds.
- **Every point must belong somewhere.** There is no "none of the above". DBSCAN (06.5) can label points as noise.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: run K-Means on the eight customers again, but start the centroids on **G (7, 8)** and **H (9, 7)**. How many iterations does it take? Do you reach the same final clusters?
2. By hand: compute the inertia if you put {A, B, C} in one cluster and {D, E, F, G, H} in the other, with centroids at their means. Is it higher or lower than 18.5? What does that tell you?
3. In code: run K-Means with $k$ = 1 to 8 on the eight customers and print the inertia for each. What is the inertia at $k = 8$, and why?
4. In code: cluster the real RFM table with `StandardScaler` but **without** the log. Compare the cluster sizes with the table in section 7.

**Quiz**

1. What are the two steps K-Means repeats?
2. What is a centroid, and does it have to be one of the customers?
3. Why does K-Means always stop?
4. Why might two runs of K-Means give different answers, and how do you guard against it?
5. Why did unscaled RFM produce one cluster of 4,300 customers?

## 10. Recap

- K-Means repeats **assign** (nearest centroid) and **update** (centroid = mean of its points) until nothing changes.
- **Inertia** = total squared distance from points to their own centroid; it only ever falls, so the algorithm stops. On the eight customers: 107 → 24.18 → 18.5.
- The answer depends on the **start**: use k-means++ and `n_init=10`.
- The answer depends even more on **scale**: log skewed features and standardise, or one feature decides everything.
- K-Means makes round clusters and forces every point into one; later lessons relax both.

**Next:** [06.3 How many clusters?](../03-choosing-k/): the elbow, the silhouette, and turning clusters into personas marketing can use.
