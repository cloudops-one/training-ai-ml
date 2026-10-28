# 04.6 · Feature engineering: dates, ratios, bins, RFM

⏱ about 1.5 hours · Prerequisites: 01.6 (groupby), 03.4 (logistic regression), 04.2 (leakage) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's churn model from Module 03 uses five columns: orders, spend, products, tenure and days since the last order. The retention manager reads the list and says: *"You're missing the obvious stuff. Customers who suddenly stop ordering. Big-basket trade buyers versus people buying one gift. Foreign customers. Did nobody think about any of that?"*

They are right that the transactions hold more than five numbers' worth of information. But a model cannot read 400,000 transaction rows about a customer. It can only read the **columns you give it**, one row per customer. Turning raw history into informative columns is called **feature engineering**, and on business data it is often where most of a model's quality comes from.

## 2. The idea

A doctor does not diagnose from your raw heartbeat recording. They compute a few summaries: resting rate, variability, any pauses. Each summary throws away detail and keeps something meaningful.

Feature engineering is the same move. You take raw facts (timestamps, amounts, product codes) and compute summaries a model can learn from:

```
raw transactions                        one row per customer
───────────────────────                 ──────────────────────────────────────────
A  2011-08-25  £40                      customer  R(days)  F  M(£)  avg basket  …
A  2011-08-10  £60          ──────►     A          7       4  180   45
A  2011-07-02  £50                      B         73       2  500   250
…                                       C        184       4  100   25
```

Four families of features cover most business problems:

| Family | Example | Why it helps |
|---|---|---|
| **Date parts** | day of week, month, days since X | A raw date is just a growing number; the meaning is in "Friday", "December", "60 days ago" |
| **Ratios** | £ per order, products per order, orders per month | Separates *how often* from *how much*; compares customers of different ages fairly |
| **Bins** | recency band 0–30 / 31–90 / 91–180 / 181+ days | Makes relationships visible; lets a linear model handle a bend |
| **Transforms** | log(1 + spend) | Pulls in extreme values so a few whales do not dominate |

## 3. Definition

A **feature** is any column the model receives as input. **Feature engineering** is creating new features from existing data, using knowledge of the problem.

**RFM** is the oldest customer-scoring method in retail, used since the 1990s mail-order catalogues:

- **Recency (R)**: days from the customer's last order to the reference date. Small is good.
- **Frequency (F)**: number of orders in the period.
- **Monetary (M)**: total amount spent in the period.

The **reference date** (cut-off) is the moment you are predicting from. Every feature must be computed only from data *before* it. A feature that uses data after the cut-off is leakage (lesson 04.2), however useful it looks.

## 4. Formula

With cut-off date $T$ and a customer's orders at dates $d_1, \dots, d_F$ with amounts $a_1, \dots, a_F$:

$$R = T - \max_i d_i \qquad F = \text{number of orders} \qquad M = \sum_i a_i \qquad \text{avg basket} = \frac{M}{F}$$

**Log transform** for a skewed amount:

$$x' = \log(1 + x)$$

- The $1 +$ part makes $x = 0$ map to $0$ instead of $-\infty$. NumPy calls it `np.log1p`.
- It squeezes large values: £100 → 4.62, £1,000 → 6.91, £100,000 → 11.51. Each ×10 becomes roughly +2.3.
- It keeps the order of values. A customer who spent more still has a bigger $x'$.

## 5. By hand

Ten transactions, cut-off **2011-09-01**:

| customer | date | amount (£) |
|---|---|---|
| A | 2011-08-25 | 40 |
| A | 2011-08-10 | 60 |
| A | 2011-07-02 | 50 |
| A | 2011-05-15 | 30 |
| B | 2011-06-20 | 300 |
| B | 2011-03-11 | 200 |
| C | 2011-03-01 | 20 |
| C | 2011-02-14 | 25 |
| C | 2011-01-20 | 15 |
| C | 2010-12-12 | 40 |

**Step 1: last and first order per customer.** A: 2011-08-25 and 2011-05-15. B: 2011-06-20 and 2011-03-11. C: 2011-03-01 and 2010-12-12.

**Step 2: recency.** Count days from the last order to 1 September.

- A: 25 Aug → 1 Sep = **7** days.
- B: 20 Jun → 1 Sep = 10 (rest of June) + 31 (July) + 31 (August) + 1 = **73** days.
- C: 1 Mar → 1 Sep = 30 + 30 + 31 + 30 + 31 + 31 + 1 = **184** days.

**Step 3: frequency and monetary.** A: F = 4, M = 40 + 60 + 50 + 30 = **180**. B: F = 2, M = **500**. C: F = 4, M = **100**.

**Step 4: ratio.** Average basket = M / F. A: 180 / 4 = **£45**. B: 500 / 2 = **£250**. C: 100 / 4 = **£25**.

**Step 5: log transform of M.** A: ln(181) = **5.198**. B: ln(501) = **6.217**. C: ln(101) = **4.615**. B spent 5× what C did, but on the log scale the gap is only 1.6.

| customer | R | F | M | avg basket | log(1+M) | story |
|---|---|---|---|---|---|---|
| A | 7 | 4 | 180 | 45 | 5.198 | a regular |
| B | 73 | 2 | 500 | 250 | 6.217 | big, rare, going quiet |
| C | 184 | 4 | 100 | 25 | 4.615 | probably gone |

Notice that A and C have the **same F** but completely different futures. Frequency alone would call them equal; recency separates them.

## 6. In code

The whole table above is one `groupby`:

```python
rfm = tx.groupby("customer").agg(last=("date", "max"), F=("date", "count"), M=("amount", "sum"))
rfm["R"] = (cutoff - rfm["last"]).dt.days
rfm["avg_basket"] = rfm["M"] / rfm["F"]
rfm["log_M"] = np.log1p(rfm["M"])
```

Date parts come from the `.dt` accessor: `.dt.day_name()`, `.dt.month`, `.dt.dayofweek`. Bins come from `pd.cut` (you choose the edges) or `pd.qcut` (equal-sized groups).

## 7. On real data

The notebook builds these features for the 3,317 real customers from Modules 01.9 and 03 and asks one question: **does the churn model get better?**

| Feature set (scaled logistic regression) | Test AUC |
|---|---|
| raw 5 features (Module 03) | 0.741 |
| same 5, skewed ones logged | 0.747 |
| + ratios (avg basket, orders per month) | 0.747 |
| + recent activity (orders in last 3 months) | 0.742 |
| + country and weekend share | 0.742 |

Two honest lessons. The log transform helped a little, because `spend` is extremely skewed (skew 16.3 before, 0.39 after) and a linear model handles the logged version better. And **adding more features did not keep helping**: the last two sets scored slightly lower. Every extra column is another chance to fit noise. The differences are also small enough to be luck on one test split; lesson 04.7 shows how to check.

The recency bands tell a clear story on their own: churn rises from 25% (ordered in the last 30 days) to 38%, 56% and 66% (silent for 181+ days).

## 8. Where it breaks

- **Leakage through features.** "Orders in September" predicts September churn perfectly, because it *is* the answer. Always ask: *would I know this value on the cut-off date?* Exercise 4 makes you build this mistake on purpose.
- **Division by zero.** Ratios need a non-zero denominator. `orders` is at least 1 here by construction; that is not always so.
- **More is not better.** Ten weak features can make a model worse than three strong ones (Lesson 04.10 shows how regularisation helps).
- **Small gains can be noise.** A 0.006 AUC lift on 830 customers is within the wobble of a single split. Use cross-validation before you celebrate.
- **Transforms matter to some models only.** Logs and scaling change linear models and KNN; trees only care about the order of values and barely notice (notebook section 9).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: customer D bought on 2011-08-31 (£12), 2011-08-01 (£18) and 2011-07-01 (£30). Compute R, F, M, average basket and log(1 + M) with cut-off 2011-09-01.
2. Bin real `spend` into quartiles with `pd.qcut` and show churn rate per quartile.
3. Create `products_per_order` and describe what a high value means for the business.
4. Build the leaky feature `orders_in_september` and see what it does to AUC. Explain why it is useless.

**Quiz**

1. What is customer C's recency, and how did you compute it?
2. Why can a model not use a raw date column well?
3. What does `np.log1p` compute, and why `1 +`?
4. Name one way feature engineering can go wrong.
5. Why do logs help logistic regression but barely affect a decision tree?

## 10. Recap

- A model only sees the columns you give it. **Feature engineering** turns raw history into informative columns.
- **RFM** = recency, frequency, monetary. Compute all three from transactions with one `groupby`.
- Date parts, ratios, bins and log transforms cover most business features.
- Every feature must be computable **at the cut-off date**. Otherwise it is leakage.
- Measure, do not assume: on ShopWave, logs gave a small lift; extra features gave none.

**Next:** [04.7 Cross-validation](../07-cross-validation/) tells you whether a 0.006 AUC gain is real or luck.
