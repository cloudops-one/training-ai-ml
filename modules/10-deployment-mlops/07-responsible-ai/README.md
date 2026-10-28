# 10.7 · Responsible AI: fairness, privacy, explainability, governance

⏱ about 1 hour · Prerequisites: 03.6 (recall, false positives), 05.8 (explanations), 10.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Before the win-back campaign goes out, ShopWave's legal and customer teams ask four questions:

1. *Does the model treat some customers worse than others?*
2. *What personal data does it use, and where does that data go, including into the chatbot's LLM?*
3. *Can we explain a decision to a customer who asks?*
4. *Who is responsible if it goes wrong, and how would we know?*

"It has a good AUC" answers none of them.

## 2. The idea

Treat a model like any other product that affects people: measure its effect on different groups, use as little personal data as the task needs, be able to explain its decisions, and write down what it is for, what it is not for, and who watches it.

Most of this is not new technique. It is the metrics from Module 03, computed **per group**, plus honesty about uncertainty and a one-page document.

## 3. Definition

- **Group fairness**: comparing a model's behaviour across groups of people.
  - **Demographic parity**: each group is flagged at the same rate.
  - **Equal opportunity**: each group's actual positives (churners) are caught at the same rate, that is, equal **recall**.
  - **Equalised odds**: equal recall **and** equal false positive rate.
- **Calibration by group**: the average prediction matches the actual rate in each group.
- **Personal data** (GDPR): any information relating to an identifiable person. **Pseudonymised** data (an ID instead of a name) is still personal data.
- **Data minimisation**: collect and use only what the purpose needs.
- **Model card**: a short document stating purpose, out-of-scope uses, data, performance, fairness, limits and ownership.
- **Human oversight**: a person can review, override or stop the system's decisions.
- **EU AI Act**: EU law that sorts AI systems into risk tiers with obligations rising with the risk.

## 4. Formula

For group $g$, with $\hat{y}$ = flagged and $y$ = churned:

$$\text{flag rate}_g = P(\hat{y}=1 \mid g) \qquad \text{recall}_g = P(\hat{y}=1 \mid y=1, g) \qquad \text{FPR}_g = P(\hat{y}=1 \mid y=0, g)$$

Demographic parity difference $= \text{flag rate}_A - \text{flag rate}_B$; equal opportunity difference $= \text{recall}_A - \text{recall}_B$.

These definitions **conflict**. When groups have different churn rates, a model generally cannot satisfy demographic parity and equalised odds at once. Choosing which one matters is a business and ethical decision, not a technical one.

## 5. By hand

Ten customers. Group A has six, group B four.

| group | flag | churned |
|---|---|---|
| A | 1, 1, 1, 0, 0, 0 | 1, 1, 0, 1, 0, 0 |
| B | 1, 0, 0, 0 | 1, 1, 0, 0 |

- **Flag rate**: A 3/6 = **0.50**; B 1/4 = **0.25**. Demographic parity difference 0.25.
- **Recall**: A's churners are customers 1, 2, 4; flagged 2 of 3 = **0.67**. B's churners are 1, 2; flagged 1 of 2 = **0.50**. Equal opportunity difference 0.17.
- **False positive rate**: A's stayers 3, 5, 6; flagged 1 of 3 = **0.33**. B's stayers 3, 4; flagged 0 of 2 = **0**.

B's churners are caught less often (bad if the voucher is a benefit), and B's loyal customers are never bothered (good if it is a nuisance). Which gap matters depends on what the decision does to people.

## 6. In code

```python
for name, g in test.groupby("group"):
    flag, churn = g["p"] >= threshold, g["churned"] == 1
    print(name, flag.mean(), (flag & churn).sum() / churn.sum(), (flag & ~churn).sum() / (~churn).sum())
```

and a bootstrap for uncertainty: resample a group with replacement 1,000 times, recompute recall, take the 2.5th and 97.5th percentiles.

## 7. On real data

ShopWave holds no data on age, gender or ethnicity, and should not. The only attribute is country, so the notebook compares **UK (749 test customers)** with **non-UK (81)**, as an example of a group the business might want treated alike.

| threshold | group | flag rate | recall | FPR | avg predicted | actual churn |
|---|---|---|---|---|---|---|
| 0.07 (£5 voucher) | UK | 0.935 | 0.997 | 0.885 | 0.432 | 0.443 |
| | non-UK | 0.951 | 1.000 | 0.922 | 0.426 | 0.370 |
| 0.35 (£20 voucher) | UK | 0.658 | 0.846 | 0.508 | 0.432 | 0.443 |
| | non-UK | 0.617 | 0.800 | 0.510 | 0.426 | 0.370 |

- At £5 nearly everyone gets a voucher: no room for unfairness.
- At £20: flag rates differ by 4 points, recall by 5, FPR by almost nothing. Calibration differs a little more: non-UK churn is over-predicted (0.43 vs 0.37).
- **But non-UK has only 30 churners.** Its recall's bootstrap 95% interval is **0.65 to 0.93**, and the UK's 0.85 sits inside it. There is no evidence of a gap, and too little data to rule out a moderate one. Keep measuring.

**Privacy.** The transactions contain no names or emails; the Customer ID is a pseudonym. But 17 countries have fewer than 5 customers (30 customers in all), and a customer who is one of two from their country, with exact order dates, could be identifiable. It is personal data. Support emails (Module 09) contain names, emails, phones and addresses. A simple pattern-based `redact()` removes the email, phone, postcode and order number before text goes to an LLM, but **not the name or street address**: patterns are not enough, and the first question is whether the LLM needs the text at all.

**Model card.** The notebook generates one from the bundle and the fairness results, including what the model must **not** be used for.

### The EU AI Act in one table (not legal advice)

| risk tier | examples | what it means |
|---|---|---|
| unacceptable | social scoring, manipulative systems | banned |
| high | credit scoring, hiring, access to essential services, insurance pricing | risk management, data governance, documentation, human oversight, registration |
| limited | chatbots, generated content | transparency: tell people they are talking to an AI |
| minimal | spam filters, recommendations, **marketing vouchers** | no specific obligations; voluntary codes |

ShopWave's churn model, used for vouchers, is minimal risk. Use the **same score** to refuse refunds or set prices and it moves up. The ShopWave support chatbot (Module 09) is limited risk: customers must be told it is an AI. Purpose decides the tier, not the algorithm.

## 8. Where it breaks

- **"We don't use protected attributes, so we're fair."** Other features can stand in for them (postcode for ethnicity, product choices for gender). Measure outcomes by group where you lawfully can.
- **Small groups.** Thirty churners give an interval 0.28 wide. Report uncertainty; never declare a model fair or unfair on a handful of people.
- **One fairness number.** The definitions conflict. State which one you chose and why.
- **Pseudonymous = anonymous.** It is not. Treat IDs with order histories as personal data.
- **Sending raw text to hosted LLMs.** Minimise and redact first; check the provider's data-retention terms.
- **A model card nobody reads.** Tie it to the release: no card, no deployment; review it each quarter.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: flag rate, recall and FPR for two new groups; which fairness definition is violated more?
2. Compare German and French customers. What can you conclude honestly?
3. Choose per-group thresholds that give 80% recall in each group. What happens, and would you deploy them?

**Quiz**

1. What is the difference between demographic parity and equal opportunity?
2. Why can't we conclude anything from the non-UK recall of 0.80?
3. Is a customer ID without a name anonymous data?
4. What should happen to a support email before it is sent to a hosted LLM?
5. Which EU AI Act tier is the voucher model in, and what would move it up?

## 10. Recap

- Fairness = Module 03's metrics **per group**: flag rate (demographic parity), recall (equal opportunity), FPR (equalised odds), plus calibration.
- The definitions conflict; choosing one is a decision about what the model does to people.
- **Small groups need uncertainty**: bootstrap intervals, not single numbers.
- Pseudonymised data is still personal; **minimise and redact** before data leaves your systems, especially to LLMs.
- Write a **model card** with purpose, misuse, data, performance, fairness, limits, oversight and owner. The **purpose**, not the algorithm, sets the regulatory risk.

**Next:** [10.P Mini-project: deploy the churn model and the chatbot](../project/)
