# Module 06 · Patterns Without Labels

**ShopWave asks:** *"Group our customers so marketing can talk to each group differently. Find fraud. What sells together?"*

**Time:** about 14 hours. **Prerequisites:** Module 01 (Pandas, statistics), Module 04 (scaling, 04.5) and the idea of distance from KNN (03.2).

## Where you are

Until now every problem came with an answer column. Churned or not. Spend next quarter. You learned a function from features to that answer.

This week's requests have no answer column at all:

- Marketing wants **four or five customer groups** with a name and a message each. Nobody has labelled customers "champion" or "drifting"; the groups have to be *discovered*.
- Finance wants to know which orders and customers look **unusual**, before anyone has confirmed a single fraud case.
- The merchandising team wants to know **which products are bought together**, so they can bundle them and place them side by side.
- And the data team keeps saying "we have too many columns to look at".

This is **unsupervised learning**: finding structure in data that has no target. It is harder to judge than supervised learning (there is no accuracy to compute), so this module is as much about *checking* results as producing them.

## What you will be able to do afterwards

- Say what unsupervised learning can and cannot tell you, and how to judge a result with no labels.
- Run K-Means by hand on eight customers: distances, assignments, new centroids, inertia, until it stops moving.
- Choose the number of clusters with the elbow, silhouette and Davies-Bouldin, then overrule them with business sense, and turn clusters into named personas.
- Build a hierarchy of customers with agglomerative clustering and read a dendrogram.
- Find clusters of any shape and flag noise with DBSCAN, and give each customer a *probability* of belonging to each group with Gaussian mixtures.
- Compress many features into a few with PCA, and draw high-dimensional data in 2-D.
- Flag anomalies with z-scores, IQR and Isolation Forest.
- Find products bought together with support, confidence and lift, and build simple recommenders.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 06.1 | [Unsupervised learning: what you can do without a target](01-what-is-unsupervised/) | 0.5 h |
| 06.2 | [K-Means by hand](02-kmeans-by-hand/) | 2 h |
| 06.3 | [How many clusters? Elbow, silhouette, and profiling clusters](03-choosing-k/) | 1.5 h |
| 06.4 | [Hierarchical clustering and dendrograms](04-hierarchical/) | 1 h |
| 06.5 | [DBSCAN and Gaussian mixtures](05-dbscan-gmm/) | 1.5 h |
| 06.6 | [PCA: fewer dimensions, same story](06-pca/) | 2 h |
| 06.7 | [t-SNE and UMAP for pictures of high-dimensional data](07-tsne-umap/) *(optional)* | 0.5 h |
| 06.8 | [Anomaly detection: z-score, IQR, Isolation Forest](08-anomaly-detection/) | 1.5 h |
| 06.9 | [Market basket analysis: support, confidence, lift](09-market-basket/) | 1.5 h |
| 06.10 | [Recommender systems: content-based and collaborative filtering](10-recommenders/) | 1 h |
| 06.P | [Mini-project: RFM customer segments and fraud flags](project/) | 1 h |

## The data in this module

- **By hand:** tiny tables typed into the notebooks: 8 customers for K-Means, 5 for hierarchical clustering, 10 for DBSCAN.
- **Real:** an **RFM table** of all 4,338 ShopWave customers, built from the cleaned 2010–2011 transactions as of 10 December 2011 (the day after the last order):
  - **R**ecency: days since the customer's last order,
  - **F**requency: number of distinct orders,
  - **M**onetary: total spent, in pounds.

  All three have long right tails (one customer spent £280,206), so lessons log-transform and standardise them before clustering. You will see in 06.2 what happens if you forget.

## The maths in this module

Only two ideas, both from earlier modules:

- **Distance** (03.2): $d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots}$. Every clustering method here is a different way of using it.
- **The mean** (01.8): a K-Means centroid is just the mean of the points in its cluster.

New formulas (inertia, silhouette, linkage, density, support/confidence/lift) are each computed by hand once, on a handful of points, before any code.
