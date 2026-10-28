# 06.3 · How many clusters? Elbow, silhouette, and profiling clusters

⏱ about 1.5 hours · Prerequisites: 06.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

K-Means needs you to say how many clusters you want. Marketing asks, reasonably: *"Why four groups? Why not three, or seven?"*

You cannot answer "because inertia is lowest at four". Exercise 3 of 06.2 showed that inertia always falls as you add clusters, all the way to zero when every customer is their own cluster. You need a better argument, and then you need to turn whatever you choose into something marketing can use: groups with names, sizes and a reason to treat them differently.

## 2. The idea

Think of packing a suitcase into compartments. One compartment: everything jumbled. Two: shoes apart from clothes, a big improvement. Three: a third for toiletries, still useful. By the tenth compartment you are separating left socks from right socks. Each extra compartment helps less, and at some point the extra effort buys you almost nothing.

Three tools put numbers on that feeling:

- **Elbow**: plot inertia against $k$ and look for where the curve bends from steep to flat.
- **Silhouette**: for each point, is it much closer to its own cluster than to the next one? Average over all points.
- **Davies-Bouldin**: for each cluster, how does its spread compare with its distance to the most similar other cluster?

Then comes the step no metric can do: **can the business name each group and act on it differently?**

## 3. Definition

- **Elbow method**: compute inertia for $k = 1, 2, \dots$ and choose the $k$ after which adding a cluster stops reducing inertia much. Visual and approximate.
- **Silhouette score** (per point, between −1 and 1): close to 1 = well inside its cluster; near 0 = on the border between two clusters; negative = probably in the wrong cluster. The **average** silhouette over all points summarises a clustering; higher is better.
- **Davies-Bouldin index** (≥ 0): lower is better. 0 would mean perfectly tight clusters infinitely far apart.
- **Cluster profile**: a table of each cluster's size and average feature values, used to describe it.
- **Persona**: a profile with a name and a sentence ("champions: buy often, buy a lot, bought recently") and a planned action.

## 4. Formula

**Silhouette of one point** $i$:

$$s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)}$$

- $a(i)$: average distance from $i$ to the **other points in its own cluster** (how snug it is at home).
- $b(i)$: average distance from $i$ to the points of the **nearest other cluster** (how far the neighbours are).

If $a$ is much smaller than $b$, $s$ is close to 1. If $a \approx b$, $s \approx 0$. If $a > b$, $s$ is negative: the point would be happier next door.

**Davies-Bouldin**, with $s_j$ = average distance from cluster $j$'s points to its centroid and $d_{jl}$ = distance between centroids $j$ and $l$:

$$DB = \frac{1}{k}\sum_{j=1}^{k} \max_{l \ne j} \frac{s_j + s_l}{d_{jl}}$$

For every cluster, find its "most confusable" neighbour (large spreads, close centres), then average.

## 5. By hand

Use the final two clusters from 06.2: cluster 1 = {A (1, 1), B (1, 3), C (2, 2), D (4, 3)}, cluster 2 = {E (6, 6), F (7, 5), G (7, 8), H (9, 7)}.

**Silhouette of D (4, 3).**

$a$ (average distance to A, B, C):

| to | A | B | C |
|---|---|---|---|
| distance | $\sqrt{9+4} = 3.606$ | $\sqrt{9+0} = 3.000$ | $\sqrt{4+1} = 2.236$ |

$a = (3.606 + 3.000 + 2.236)/3 = 2.947$

$b$ (average distance to E, F, G, H):

| to | E | F | G | H |
|---|---|---|---|---|
| distance | $\sqrt{4+9} = 3.606$ | $\sqrt{9+4} = 3.606$ | $\sqrt{9+25} = 5.831$ | $\sqrt{25+16} = 6.403$ |

$b = (3.606 + 3.606 + 5.831 + 6.403)/4 = 4.861$

$$s(D) = \frac{4.861 - 2.947}{4.861} = 0.394$$

D belongs in cluster 1, but not by much: it is the customer that switched teams in 06.2. For comparison, C (right in the middle of cluster 1) scores 0.758. Averaging all eight customers gives a silhouette score of **0.637**.

**Davies-Bouldin.** Average distance to own centroid: cluster 1 $s_1 = (1.601 + 1.250 + 0.250 + 2.136)/4 = 1.309$; cluster 2 $s_2 = (1.346 + 1.521 + 1.521 + 1.820)/4 = 1.552$. Centroids (2, 2.25) and (7.25, 6.5) are $d = \sqrt{5.25^2 + 4.25^2} = 6.755$ apart.

$$DB = \frac{1.309 + 1.552}{6.755} = 0.424$$

(With only two clusters, each one's most confusable neighbour is the other, so both terms are equal.)

**Elbow.** Inertia on the eight customers for $k = 1 \dots 4$: 109.75, 18.5, 12.25, 6.17. From 1 to 2 clusters, inertia drops by 91; from 2 to 3 by only 6. A clear elbow at $k = 2$.

## 6. In code

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, km.inertia_, silhouette_score(X, km.labels_), davies_bouldin_score(X, km.labels_))
```

`silhouette_samples` gives one value per point, which is useful for spotting customers stuck between two groups.

## 7. On real data

All 4,338 customers, RFM logged and standardised:

| $k$ | inertia | silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|---|
| 2 | 6,488 | **0.434** | **0.892** |
| 3 | 4,879 | 0.338 | 1.048 |
| 4 | 3,963 | 0.337 | 1.013 |
| 5 | 3,332 | 0.316 | 0.995 |
| 6 | 2,875 | 0.314 | 1.020 |
| 7 | 2,573 | 0.309 | 0.977 |
| 8 | 2,360 | 0.302 | 0.993 |

**The metrics disagree with each other, and with common sense.** Silhouette and Davies-Bouldin both prefer $k = 2$. The elbow bends gently around 3 to 4. That is typical of real customers: they do not come in neat, separate blobs, they form one continuous cloud from "bought once, long ago" to "buys every week".

So look at what each choice *means*. With $k = 2$:

| group | customers | recency (days) | orders | spend (£) |
|---|---|---|---|---|
| active | 1,652 | 25 | 8.5 | 4,570 |
| inactive | 2,686 | 133 | 1.7 | 498 |

True, but marketing knew that already. Two groups do not support four different campaigns.

With $k = 4$, sorted by spend:

| persona | customers | recency (days) | orders | spend (£) | share of revenue | action |
|---|---|---|---|---|---|---|
| **Champions** | 699 | 11 | 13.9 | 8,194 | 64% | VIP perks, early access, never discount |
| **Regulars, cooling** | 1,178 | 70 | 4.1 | 1,815 | 24% | "we miss you" reminder before they lapse |
| **Newcomers** | 826 | 17 | 2.2 | 556 | 5% | welcome series, second-order incentive |
| **Lapsed one-timers** | 1,635 | 180 | 1.3 | 344 | 6% | cheap reactivation email, or let go |

Each row has a one-sentence description and a different action. The silhouette at $k = 4$ (0.337) is almost the same as at $k = 3$ (0.338), only 2% of customers have a negative silhouette, and 06.1 showed these four groups are stable across random seeds. **Four is the answer you can defend**: good enough on the metrics, stable, and useful.

## 8. Where it breaks

- **The elbow is often not sharp.** On real data the curve bends gradually and two people will point at different $k$. Treat it as a range, not an answer.
- **Silhouette favours few, well-separated groups.** On a continuous cloud of customers it nearly always prefers $k = 2$. A high silhouette for a useless split is still useless.
- **All three metrics assume roughly round clusters** (they are built on distances to centres or averages). For odd shapes, use DBSCAN (06.5) and judge by eye.
- **Profiles use averages.** A mean of £8,194 for champions hides a few customers with £100,000+. Check medians too.
- **Segments drift.** Customers move between groups every month. Re-run, and track how many move.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: compute the silhouette of customer **E (6, 6)** in the two-cluster solution. Then check all eight with `silhouette_samples`.
2. Plot the silhouette of every real customer for $k = 4$, grouped by persona. Which persona has the most customers with a negative silhouette? Why might that be?
3. Profile $k = 5$ with means **and medians**. Which persona splits, and would marketing treat the two halves differently?
4. Write a one-paragraph email to the marketing director recommending a number of segments. Use at least one metric and at least one business argument.

**Quiz**

1. Why can you not choose $k$ by picking the lowest inertia?
2. In the silhouette formula, what do $a(i)$ and $b(i)$ measure?
3. What does a negative silhouette tell you about a point?
4. For Davies-Bouldin, is higher or lower better?
5. Both silhouette and Davies-Bouldin preferred $k = 2$ on ShopWave customers. Why did we choose 4?

## 10. Recap

- Inertia always falls with $k$; look for the **elbow** where it stops falling fast.
- **Silhouette** $s = (b - a)/\max(a, b)$ per point; average it; higher is better. D scored 0.394, the eight customers 0.637.
- **Davies-Bouldin** compares cluster spread with distance between centres; lower is better.
- On real customers the metrics disagree and favour $k = 2$; **profiles and business sense** chose 4 personas: Champions (699, 64% of revenue), Regulars cooling (1,178), Newcomers (826), Lapsed one-timers (1,635).
- A segmentation is finished only when each group has a name, a size and an action.

**Next:** [06.4 Hierarchical clustering](../04-hierarchical/): instead of choosing $k$ up front, build the whole family tree of customers and cut it where you like.
