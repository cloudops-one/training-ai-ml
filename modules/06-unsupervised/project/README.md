# 06.P · Mini-project: RFM customer segments and fraud flags

⏱ about 1 hour (longer if you write a careful summary) · Prerequisites: all of Module 06 · Notebooks: [project.ipynb](project.ipynb) (your work), [solution.ipynb](solution.ipynb) (one complete answer)

---

## The brief

Two teams at ShopWave have asked for your help in the same week.

**Marketing** wants to stop sending the same email to all 4,338 customers. They ask: *"Split our customers into a handful of groups we can treat differently, tell us who is in each, and what we should do with each."*

**Finance** wants a short, regular list of orders worth a second look. They ask: *"We can review about 15 orders a month. Which ones?"*

There are no labels for either question. Nobody has tagged customers as "loyal" or orders as "suspicious". This is unsupervised learning end to end.

## What to deliver

A notebook ([project.ipynb](project.ipynb)) that runs top to bottom and contains:

1. **RFM table**: recency, frequency, monetary per customer as of 2011-12-10, log-transformed and scaled.
2. **Choice of k**: elbow and silhouette for k = 2 to 8, plus a written justification that uses both the numbers and business sense.
3. **Segment profiles**: size, median R/F/M and share of revenue per segment, with business names assigned by code rules.
4. **A picture**: PCA to 2-D, coloured by segment, with an interpretation of PC1.
5. **An anomaly list**: Isolation Forest on product-only invoices (amount, items, average price per item, hour), with the top 10 shown and explained.
6. **Flags vs segments**: do odd invoices come mostly from certain segments?
7. **A one-page summary for marketing and finance** in plain language, with numbers.

## Hints

- Reuse code from the lessons: RFM from 06.2–06.3, PCA from 06.6, Isolation Forest from 06.8.
- Remove non-product lines (manual adjustments, postage, fees) before the anomaly step. Lesson 06.8 showed why.
- Cluster numbers from K-Means are arbitrary and can change between runs. Name segments with rules on their profiles (for example "highest median spend = Champions").
- If the metrics prefer a k that is not useful to the business, say so and explain your choice. That is a normal, honest result.
- Write the summary for someone who has never heard of K-Means.

## Rubric (20 points)

| Area | Points | Full marks when… |
|---|---|---|
| RFM table | 2 | correct definitions and snapshot date; log + scale applied; shape described |
| Choosing k | 3 | both plots shown; choice justified with numbers *and* business reasoning |
| Profiles and names | 4 | sizes, medians and revenue shares shown; names assigned by rule; each name fits its profile |
| PCA picture | 2 | readable plot, labelled axes and legend; PC1 interpreted from its weights |
| Anomaly list | 4 | non-product lines removed; sensible features, logged; top 10 shown with a plausible explanation |
| Flags vs segments | 2 | comparison against each segment's share of all invoices, not raw counts; honest conclusion |
| Summary | 3 | one action per segment, review-list workload stated, numbers quoted, no jargon |

## Stretch goals

- Re-run the segmentation on 2010 data only and see how many customers stay in the same segment a year later.
- Replace K-Means with a Gaussian mixture (06.5) and report the customers whose segment is uncertain (no probability above 0.6).
- Use the recommender from 06.10 to suggest three products for one Champion and one Newcomer.

## What the solution found

Look at [solution.ipynb](solution.ipynb) only after finishing. In short: four segments, with **Champions** (699 customers, 16%) bringing in **64%** of revenue and **Lapsed one-timers** (1,635 customers, 38%) only 6%. Silhouette preferred k = 2; k = 4 was chosen for usefulness. After removing 1,543 non-product lines, Isolation Forest flags **185** invoices a year, about **15 a month**, mostly orders of one or two expensive items. Perhaps surprisingly, flagged invoices come from every segment roughly in proportion to its order volume, not mainly from new customers.
