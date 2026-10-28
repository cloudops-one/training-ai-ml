# 06.8 · Anomaly detection: z-score, IQR, Isolation Forest

⏱ about 1.5 hours · Prerequisites: 01.8 (mean, std, percentiles), 05.2 (random forest idea) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's finance team reviews odd orders by hand: possible fraud, typing errors, refunds booked as sales. They have time for about 180 invoices a year out of 18,500. Which 180 should they look at?

There are no labels. Nobody has marked past invoices as "bad", so the supervised methods from Modules 02–05 have nothing to learn from. You need a method that says *"this one does not look like the others"* without being told what "bad" looks like.

## 2. The idea

Two ways to spot the odd one out:

- **Distance from the centre.** How far is this value from the typical value, measured in typical spreads? That is the z-score, and its sturdier cousin, the IQR rule.
- **Ease of isolation.** Play a game of "guess the number" with random yes/no cuts. An odd value gets separated from the crowd in one or two cuts; a typical value hidden in the middle of the pack needs many. That is the idea behind **Isolation Forest**, and it works with many features at once.

## 3. Definition

- **Anomaly (outlier)**: a data point that differs markedly from the rest. It might be an error, fraud, or simply rare and genuine.
- **z-score**: how many standard deviations a value sits from the mean.
- **IQR rule**: flag values more than 1.5 × IQR below the first quartile or above the third.
- **Masking**: an extreme value inflates the mean and std so much that its own z-score looks normal.
- **Isolation Forest**: an ensemble of random trees that split on random features at random cut points; points that end up alone after few splits get high anomaly scores.
- **Contamination**: the share of points you choose to label as anomalies (the cut-off on the score).
- **Local Outlier Factor (LOF)**: compares a point's local density with its neighbours' densities; much sparser than its neighbours means anomalous.

## 4. Formula

$$z = \frac{x - \bar{x}}{\sigma} \qquad \text{flag if } |z| > 3$$

$$\text{IQR} = Q_3 - Q_1 \qquad \text{flag if } x < Q_1 - 1.5\,\text{IQR} \ \text{ or } \ x > Q_3 + 1.5\,\text{IQR}$$

**Isolation Forest score** for a point $x$, with $h(x)$ the number of splits needed to isolate it in one tree, $E[h(x)]$ its average over all trees, and $c(n)$ the average path length for a sample of $n$ points (a normalising constant):

$$s(x) = 2^{-E[h(x)] / c(n)}$$

Short paths give scores near 1 (anomaly); typical paths give about 0.5. scikit-learn's `score_samples` returns the negative of this, so we flip the sign.

## 5. By hand

Ten ShopWave orders, in pounds: 42, 55, 38, 61, 47, 52, 44, 58, 49, **410**.

**z-scores.** Mean = 856 / 10 = **85.6**. Population std:

$$\sigma = \sqrt{\tfrac{1}{10}\sum (x - 85.6)^2} = 108.3$$

For £410: $z = (410 - 85.6) / 108.3 = 2.99$. **Not flagged** at 3. The £410 order dragged the mean from about £50 up to £85.60 and inflated the std, hiding itself. That is masking.

**IQR.** Sorted: 38, 42, 44, 47, 49, 52, 55, 58, 61, 410. With NumPy's quartiles, $Q_1 = 44.75$ and $Q_3 = 57.25$, so IQR = 12.5. Upper fence = 57.25 + 1.5 × 12.5 = **76**. £410 is far above it: **flagged**. Quartiles barely move when one value goes wild.

**Random cuts.** Pick a cut uniformly between the smallest (38) and largest (410) value. The chance it falls between 61 and 410, which isolates £410 straight away, is (410 − 61) / (410 − 38) = 349 / 372 = **94%**. A middle value such as £49 needs about five cuts on average. Averaged over 1,000 random runs in the notebook: £410 needs 1.05 cuts, the others 3.5 to 5.3.

## 6. In code

```python
from sklearn.ensemble import IsolationForest
X = np.log1p(inv[["amount", "items", "avg_price", "hour"]])   # log skewed columns
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(X)
inv["score"] = -iso.score_samples(X)      # higher = odder
inv["flag"] = iso.predict(X) == -1        # -1 = anomaly
```

LOF for a second opinion:

```python
from sklearn.neighbors import LocalOutlierFactor
flag = LocalOutlierFactor(n_neighbors=20, contamination=0.01).fit_predict(X) == -1
```

## 7. On real data

18,532 invoices from 2011, one row per invoice: amount, items, average price per item, hour. With `contamination=0.01`, Isolation Forest flags **186**.

**What it found.** The top of the list is not shady customers. It is mostly:

- **Manual adjustments** (stock code `M`) and **POSTAGE** charges, which are not product sales at all;
- single expensive furniture pieces (a sideboard, a kitchen cabinet) in a shop that mostly sells £2 gifts.

**44%** of the flagged invoices are non-product lines (manual, postage, bank charges). That is a data-quality finding: these lines should be removed before computing revenue per customer. With them removed (exercise 2), the list turns into genuine but unusual customer orders: a few items, high prices, sometimes late in the evening.

**Customers vs invoices.** Lesson 06.5 already met anomalies at the *customer* level: DBSCAN left 139 customers as noise, with a median spend of £7,205. Here we look at *invoices*, a different unit with different odd cases. Always ask "odd what?" before choosing features.

**The biggest order is not the oddest.** The £168,470 order for 80,995 paper craft kits ranks only **95th**. Its price per item (£2.08) is perfectly normal; only its size is extreme. If you care about big orders, a simple rule is clearer.

**Contamination only moves the cut-off**: 0.5% flags 93 invoices, 1% flags 186, 5% flags 927. Choose it from how many cases the team can review.

**Methods disagree.** LOF also flags 186 invoices, but only 28 are on both lists. Each method has its own idea of "odd".

**Stability.** Changing Isolation Forest's random seed keeps 82–86% of the flags. The core of the list is stable; its edge is noise.

## 8. Where it breaks

- **Anomalous is not bad.** The detector finds unusual shapes; a human decides whether they are fraud, errors or great customers.
- **No labels, no metrics.** You cannot compute precision or recall. Read the top of the list, check stability, and get a small labelled sample from the business.
- **z-scores mask.** Extreme values hide themselves; use IQR, median/MAD (exercise 1) or Isolation Forest.
- **Features decide what counts as odd.** Leave out `hour` and late-night orders can never stand out. Choose features with the people who will review the flags.
- **Skewed features need logs.** Without them, amounts in the thousands dominate everything.
- **Contamination is a guess.** It sets workload, not truth.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Compute the robust (median/MAD) z-score for the £410 order by hand.
2. Remove non-product lines first, then re-run the detector. What rises to the top?
3. Compare feature medians of flagged vs all invoices. What makes an invoice "odd" here?

**Quiz**

1. What does a z-score measure?
2. Why did the £410 order escape the z-score rule but not the IQR rule?
3. In one sentence, how does Isolation Forest decide a point is anomalous?
4. What does `contamination` change, and what does it not change?
5. Name two ways to evaluate an anomaly detector when you have no labels.

## 10. Recap

- Anomaly detection ranks points by how unusual they are, without labels.
- **z-scores** are simple but suffer from masking; **IQR** and median-based rules are sturdier.
- **Isolation Forest** isolates points with random cuts: few cuts means odd. It handles many features and skewed data (after a log).
- **Contamination** sets how many you flag, i.e. the reviewers' workload.
- Flags need human judgement. On ShopWave the best catch was a data-quality problem, not fraud.

**Next:** [06.9 Market basket analysis](../09-market-basket/)
