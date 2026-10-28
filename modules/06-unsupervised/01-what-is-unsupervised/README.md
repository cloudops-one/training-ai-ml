# 06.1 · Unsupervised learning: what you can do without a target

⏱ about 30 minutes · Prerequisites: 00.3 (three kinds of learning), 01.8 (statistics) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Four requests land on your desk in the same week:

1. **Marketing:** "We send everyone the same email. Split our 4,338 customers into a few groups we can write different emails for."
2. **Finance:** "Some accounts look odd. Before we have confirmed any fraud, can you show us the strangest ones?"
3. **Merchandising:** "Which products end up in the same basket? We want to bundle them."
4. **The data team:** "Every customer has dozens of numbers. Can we look at them on one chart?"

Every model you have built so far needed a **target**: a column holding the right answer (churned, spend next quarter). None of these four requests has one. Nobody has labelled customers "loyal", nobody has confirmed fraud, nobody has written down which products "belong together".

Supervised learning cannot start without answers. You need tools that find structure in the data itself.

## 2. The idea

Imagine tipping a box of 500 unsorted buttons onto a table. Nobody tells you the categories, but within a minute you have piles: big and small, metal and plastic, two-hole and four-hole. You also notice the odd one, a button shaped like a star. And you notice that the brass ones always seem to come with a matching brass clasp.

You did three unsupervised things there: **grouping** (piles), **spotting oddities** (the star) and **finding associations** (brass with brass). You did a fourth without noticing: you described each button by two or three properties instead of fifty, which is **compression**.

Unsupervised learning is that, done with numbers, at scale.

## 3. Definition

**Unsupervised learning** finds structure in data that has features $X$ but **no target** $y$.

The five things it can do, each with its ShopWave request:

| task | question | ShopWave | lessons |
|---|---|---|---|
| **Clustering** | which rows are similar to each other? | customer segments for marketing | 06.2–06.5 |
| **Anomaly detection** | which rows are unlike the rest? | odd accounts for finance | 06.8 |
| **Dimensionality reduction** | can we describe rows with fewer numbers? | one chart of all customers | 06.6–06.7 |
| **Association rules** | which things occur together? | products bought together | 06.9 |
| **Recommendation** | what would this person like next? | "customers who bought this also bought…" | 06.10 |

Key words:

- **Cluster**: a group of rows that are closer to each other than to rows in other groups.
- **Segment**: a cluster that the business has given a name and an action ("champions: invite to the VIP club").
- **Anomaly / outlier**: a row far from the others.
- **Dimension**: one feature column. "Reducing dimensions" means replacing many columns with a few new ones that keep most of the information.

## 4. Formula

There is no new formula, but one important change of shape. Compare:

$$\text{supervised:}\quad \text{learn } f \text{ so that } f(X) \approx y \qquad\qquad \text{unsupervised:}\quad \text{learn structure in } X \text{ alone}$$

In scikit-learn the difference is visible in a single argument:

```python
model.fit(X, y)     # supervised: features and answers
model.fit(X)        # unsupervised: features only
```

And it changes how you judge the result. With a target you compute accuracy or AUC on test data. Without one, you can only ask three weaker questions:

1. **Is the structure tight?** Are points close to their own group and far from others? (Silhouette score, 06.3.)
2. **Is it stable?** Do you get the same groups with a different random seed, or a different sample of customers?
3. **Is it useful?** Can the business describe each group in one sentence and act on it differently?

The third question matters most. A clustering with a beautiful score that marketing cannot use is a failure.

## 5. By hand

Sort these eight requests into the five tasks from section 3 (or "supervised", if a target exists). Write your answers down before reading on.

| # | request |
|---|---|
| 1 | Find groups of products with similar sales patterns over the year. |
| 2 | Predict whether a customer will churn next quarter. |
| 3 | Flag invoices whose quantities are wildly unlike the usual. |
| 4 | Suggest three products on each customer's account page. |
| 5 | Squash 4,000 product columns (did the customer buy it, yes or no) into 10 numbers per customer. |
| 6 | Find pairs of products that appear in the same invoice far more often than chance. |
| 7 | Split customers into groups by how recently, how often and how much they buy. |
| 8 | Estimate next month's revenue. |

**Answers:** 1 clustering · 2 supervised (target: churned) · 3 anomaly detection · 4 recommendation · 5 dimensionality reduction · 6 association rules · 7 clustering · 8 supervised (target: revenue; Module 07).

The test is always the same: **is there a column of known answers to learn from?** If yes, supervised. If no, which of the five structures are you looking for?

## 6. In code

The notebook runs one tiny example of each kind on the real ShopWave customers. Do not worry about how each method works yet; each gets its own lesson. Notice the shape of the calls: `fit(X)` with no `y`.

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

segments = KMeans(n_clusters=4, random_state=42).fit_predict(X)   # clustering
pca = PCA().fit(X)                                                 # compression
pca.explained_variance_ratio_
```

## 7. On real data

The notebook builds the **RFM table** (recency, frequency, monetary) for all 4,338 customers and then:

- **Clusters** them into 4 groups. One group of 699 customers brings in 64% of all revenue.
- **Spots oddities**: 64 customers sit more than 3 standard deviations from the mean on at least one of R, F or M. The most extreme spent £280,206.
- **Compresses** the three columns: one combined direction carries 75% of the variation, so a single number already summarises most of each customer.
- **Finds an association**: across 18,532 invoices, the pink polka-dot jumbo bag and the red retrospot jumbo bag appear together in 462 of them.
- **Checks stability**: re-running K-Means with four different random seeds gives almost the same groups (agreement between 0.965 and 0.986 on a scale where 1 is identical and 0 is chance).

## 8. Where it breaks

- **There is no "correct answer" to check against.** Two analysts can produce two different segmentations from the same data and both be defensible. Judge by stability and usefulness, not just a score.
- **Garbage in, groups out.** Clustering always returns clusters, even on random noise. The algorithm never says "there is no structure here".
- **Scale decides everything.** Distance-based methods treat a £1 difference and a 1-order difference as equal unless you scale. 06.2 shows how badly this goes wrong.
- **Results need a human.** A cluster is just a set of row numbers until someone names it and decides what to do with it.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Rerun the clustering with 3 and 5 groups instead of 4. Which group from the 4-group version splits or merges?
2. Generate 1,000 rows of pure random noise with two columns and run K-Means with 4 clusters. Does it refuse? What does that tell you?
3. Find the second and third most common product pairs in the notebook's co-purchase count. Would you bundle them?

**Quiz**

1. What is the one thing supervised data has that unsupervised data does not?
2. Name the five kinds of unsupervised task.
3. In scikit-learn, how does the `fit` call differ between the two?
4. Give three ways to judge a clustering when there is no target.
5. A clustering has an excellent score but marketing cannot describe the groups. Is it a good result?

## 10. Recap

- Unsupervised learning finds structure in $X$ alone: **clusters, anomalies, compressions, associations, recommendations**.
- The tell-tale sign is `fit(X)` with no `y`.
- Without a target you judge by **tightness**, **stability** and above all **usefulness** to the business.
- Algorithms always return an answer, even on noise. Scaling and human judgement are part of the method, not extras.

**Next:** [06.2 K-Means by hand](../02-kmeans-by-hand/): the most used clustering algorithm, run step by step on eight customers until it stops moving.
