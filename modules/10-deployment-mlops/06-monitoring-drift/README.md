# 10.6 · Monitoring, data drift, concept drift, retraining

⏱ about 1.5 hours · Prerequisites: 10.1, 10.5, 03.7 (costs), 01.8 (percentiles) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The churn model is live. Every Monday it scores customers; vouchers go out. Nothing crashes, so everyone assumes it works.

But a model is a snapshot of the past. Christmas comes, a competitor opens, the website changes, and the customers the model sees stop looking like the ones it learned from. A broken model does not raise an error. It keeps producing confident numbers that are quietly wrong, and the first sign is a campaign that loses money three months later.

## 2. The idea

Watch three things, each at its own speed:

1. **Inputs** (available immediately): do this week's customers look like the training customers?
2. **Outputs** (immediately): are the model's scores distributed as they used to be?
3. **Outcomes** (late, when labels arrive): is the model still right? Both its **ranking** (AUC) and its **calibration** (are 40% predictions followed by about 40% churn?).

Then decide **by evidence** when to retrain, and on what.

## 3. Definition

- **Data drift**: the distribution of the inputs changes. Example: customers order less often in summer.
- **Concept drift**: the relationship between inputs and outcome changes. Example: 30 days without an order means more churn in January than in October.
- **Population Stability Index (PSI)**: a single number for how much a distribution moved between a reference sample and a new one.
- **Kolmogorov–Smirnov (KS) test**: the largest gap between two cumulative distributions, $D$, with a p-value.
- **Calibration gap**: actual churn rate minus average predicted churn.
- **Label delay**: the time before the truth is known. Here 91 days: a customer scored on 1 December has "churned" only if they have not ordered by 2 March.
- **Retraining policy**: the rule for when to train a new model and how to decide whether it replaces the old one.

## 4. Formula

Split the reference sample into bins (deciles); let $e_i$ and $a_i$ be the share of the reference and the new sample in bin $i$:

$$\text{PSI} = \sum_i (a_i - e_i)\,\ln\frac{a_i}{e_i}$$

Rule of thumb: **< 0.1 stable, 0.1–0.25 watch, > 0.25 act**.

KS distance between two samples with cumulative distribution functions $F_1, F_2$:

$$D = \max_x \left| F_1(x) - F_2(x) \right|$$

## 5. By hand

**PSI.** Reference: 20% of customers in each of five bins. This month: 10%, 15%, 20%, 25%, 30%.

| bin | $e$ | $a$ | $a - e$ | $\ln(a/e)$ | term |
|---|---|---|---|---|---|
| 1 | 0.20 | 0.10 | −0.10 | −0.693 | 0.0693 |
| 2 | 0.20 | 0.15 | −0.05 | −0.288 | 0.0144 |
| 3 | 0.20 | 0.20 | 0 | 0 | 0 |
| 4 | 0.20 | 0.25 | +0.05 | +0.223 | 0.0112 |
| 5 | 0.20 | 0.30 | +0.10 | +0.405 | 0.0405 |

PSI = **0.135**: "watch". Each term is positive whichever way the share moved, so shifts never cancel out.

**KS.** Last month's recencies: 10, 20, 30, 40, 50 days. This month's: 30, 40, 50, 60, 70. At $x = 20$: 40% of last month is at or below 20, 0% of this month. At $x = 50$: 100% against 60%. The largest gap is **D = 0.40**. With only five customers each, the p-value is 0.87: no evidence of a change. With 4,000 customers, the same D would be overwhelming. That is why teams alert on effect sizes (D, PSI), not p-values.

## 6. In code

```python
psi = cm.psi(train["days_since_last"], this_month["days_since_last"])     # in lib/churn_model.py
from scipy.stats import ks_2samp
D, p_value = ks_2samp(train["days_since_last"], this_month["days_since_last"])

# once labels arrive
auc = roc_auc_score(labels, scores)
gap = labels.mean() - scores.mean()
```

## 7. On real data

The notebook replays a year. It trains on customers as of **1 September 2010** (3,321 customers, 41.5% churn), using both years of ShopWave data and the production model's 274-day history window. It then scores customers on the first of every month until September 2011, exactly as production would.

| month | PSI tenure | PSI recency | PSI score | predicted churn | actual churn | AUC |
|---|---|---|---|---|---|---|
| 2010-10 | **0.877** | 0.030 | 0.009 | 0.411 | 0.459 | 0.715 |
| 2010-12 | 0.228 | 0.157 | 0.050 | 0.391 | **0.648** | 0.754 |
| 2011-01 | **0.896** | **0.415** | 0.023 | 0.403 | **0.650** | 0.771 |
| 2011-06 | **0.509** | 0.193 | 0.043 | 0.431 | 0.610 | 0.778 |
| 2011-09 | 0.023 | 0.005 | 0.006 | 0.424 | 0.436 | 0.742 |

(Orders, spend and products stayed below 0.02 all year.)

What a dashboard would have shown, and what was really happening:

- **Ranking held**: AUC between 0.715 and 0.778 all year.
- **Calibration broke**, the real problem: in December 65% churned against 39% predicted, a gap of **+0.26**. Customers seen just after their Christmas shopping did not come back in the quiet months. That is concept drift, and **no input or score alarm caught it**; only the labels did, three months later.
- **The loudest alarms were about the calendar.** Tenure "act" alerts fired most months because of where each history window starts: the training window began on 1 December 2009, in the busy pre-Christmas weeks, so **20%** of training customers sat at the 274-day tenure cap; October's window began on 31 December, with the shop closed, and only **4%** did. The model barely uses tenure, so the scores did not move (score PSI at most 0.056). Investigate before you act on an alarm.
- **Recency** really did shift in January (KS D = 0.122, median 71 → 59 days), and the p-value (9 × 10⁻²⁴) shows how meaningless p-values are with thousands of customers.

**Should we just retrain on the newest data?** On the September 2011 customers:

| model trained on | AUC | predicted | actual | cost at a £20 voucher |
|---|---|---|---|---|
| September 2010 | 0.742 | 0.424 | 0.436 | £56,560 |
| March 2011 | 0.739 | 0.607 | 0.436 | £58,120 |
| June 2011 | 0.740 | 0.593 | 0.436 | £58,300 |

Same ranking, but the newer models carry spring's high churn into September and over-predict by 16 points. The **same-season** model is best calibrated and cheapest. Recency is not always better; seasonality matters.

## 8. Where it breaks

- **Only watching inputs.** Input drift can be harmless (tenure here) and harmful drift can be invisible in the inputs (calibration here). Watch outcomes too, as soon as they arrive.
- **P-value alerts.** With thousands of rows every alert fires. Use PSI or KS D with thresholds.
- **Ignoring label delay.** Today's scores can only be judged in three months. Plan dashboards around that; look for earlier proxies (no site visit in 30 days).
- **Automatic retraining.** "Retrain every month and deploy" can make things worse, as the table shows. Retrain, then compare challenger and champion on the newest labelled month, then promote (10.5).
- **Fixing the alarm, not the cause.** A feature-definition artefact (the tenure cap) should be fixed in the feature, not by raising the alarm threshold until it stops ringing.

### A retraining policy for ShopWave

1. Every month, compute input PSI and score PSI; investigate any "act".
2. Every month, once the 91-day labels for three months ago arrive, compute AUC and the calibration gap.
3. Train a challenger on the newest labelled data **and** keep the same-season model from last year as a second candidate.
4. Promote a challenger only if it beats the champion on the newest labelled month in AUC and calibration, then re-check the threshold (10.3).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: PSI for reference 30/30/20/10/10% and this month 20/25/25/15/15%.
2. Write a calibration alert (|gap| > 0.10). In which months does it fire, and when would you have known?
3. Simulate monthly rolling retraining for June–September 2011 and compare with the fixed model.

**Quiz**

1. What is the difference between data drift and concept drift?
2. What are the three PSI bands?
3. Why can't AUC be monitored on the day you score customers?
4. Why alert on PSI or KS D rather than a p-value?
5. How can a model keep its AUC but still cost the business money?

## 10. Recap

- Monitor **inputs** and **scores** (immediately) and **outcomes** (when labels arrive): AUC for ranking, the gap for calibration.
- **PSI** $= \sum (a-e)\ln(a/e)$: < 0.1 stable, 0.1–0.25 watch, > 0.25 act. **KS D** is the largest gap between cumulative distributions.
- Alarms need investigation: ShopWave's loudest one was a feature-definition artefact; its worst problem raised no early alarm.
- Seasonality can make the newest data the wrong training data. Retrain, **compare**, then promote.

**Next:** [10.7 Responsible AI](../07-responsible-ai/)
