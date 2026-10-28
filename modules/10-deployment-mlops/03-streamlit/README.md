# 10.3 · A web app with Streamlit

⏱ about 1 hour · Prerequisites: 10.1, 03.7 (thresholds from costs) · Notebook: [lesson.ipynb](lesson.ipynb) · Program: [app.py](app.py)

---

## 1. The problem

The retention team does not write code and will not call an API. They want a page where they type in a customer's history, see the churn risk, see **why**, and try "what if the voucher cost £10 instead of £5?". And they want it this week, not after the web team's next release.

## 2. The idea

**Streamlit** turns a Python script into a web page. Every widget (a number box, a slider) is one line of Python that returns its current value. When someone changes a widget, Streamlit re-runs the script from the top and redraws the page. No HTML, no JavaScript.

The "why" comes from the model itself. Logistic regression adds up one contribution per feature before the sigmoid; showing those contributions as bars explains the score in the model's own terms.

## 3. Definition

- **Streamlit app**: a Python script that uses `st.*` functions to draw widgets and results. Started with `python -m streamlit run app.py`.
- **Rerun model**: any interaction re-executes the whole script. Expensive work (loading the model) must be cached with `@st.cache_resource` so it happens once.
- **Contribution** of feature *j*: $w_j z_j$, its weight times its standardised value. Positive pushes the log-odds (and the risk) up, negative pushes it down, relative to an average customer.
- **Break-even threshold**: the probability above which a voucher pays for itself, $t^* = c_\text{voucher} / c_\text{lost}$ (03.7).
- **AppTest**: Streamlit's testing tool; it runs the app without a browser and lets code read and change widgets.

## 4. Formula

From 10.1, the model computes

$$\text{log-odds} = b_0 + \sum_j w_j z_j, \qquad p = \frac{1}{1 + e^{-\text{log-odds}}}$$

so each feature's **contribution** is $w_j z_j$, and they add up exactly to the log-odds (with $b_0$, the log-odds of a perfectly average customer).

The voucher decision uses the break-even rule:

$$\text{send a voucher if } p \cdot c_\text{lost} > c_\text{voucher} \iff p > \frac{c_\text{voucher}}{c_\text{lost}}$$

## 5. By hand

A loyal customer: 10 orders, £2,000 spend, 60 products, 300 days since the first order, 20 since the last. Using the bundle's means, standard deviations and weights from 10.1:

| feature | $z$ | $w$ | contribution $w z$ |
|---|---|---|---|
| orders | (10 − 3.37) / 5.19 = 1.278 | −0.992 | **−1.267** |
| spend | (2000 − 1510) / 5347 = 0.092 | +0.060 | +0.005 |
| products | (60 − 48.45) / 61.67 = 0.187 | −0.450 | −0.084 |
| tenure_days | (300 − 176.08) / 78.01 = 1.589 | −0.173 | −0.274 |
| days_since_last | (20 − 92.22) / 77.41 = −0.933 | +0.437 | **−0.408** |

log-odds = −0.476 − 1.267 + 0.005 − 0.084 − 0.274 − 0.408 = **−2.504**, so $p = 1/(1+e^{2.504}) = $ **0.076**.

The bars the app draws are this last column: the number of orders is by far the biggest reason this customer is safe, then how recently they ordered.

Decision at £5 / £60: break-even $t^* = 5/60 = 0.083$. Since 0.076 < 0.083, **no voucher**: the expected loss, 0.076 × £60 = £4.54, is less than the £5 voucher.

Note the saved bundle's threshold is **0.07**, chosen on data in 05.10, and would send this customer a voucher. The two thresholds differ because the model's probabilities are not perfectly calibrated. The app uses the break-even rule because it lets the team change the costs; at the default costs the two rules disagree only for the few customers between 0.07 and 0.083.

## 6. In code

```python
import streamlit as st

@st.cache_resource
def get_bundle():
    return load_bundle()                         # once, not on every click

voucher_cost = st.slider("Voucher cost (£)", 1, 40, 5)
orders = st.number_input("Orders", min_value=1, value=2)
...
p = bundle["pipeline"].predict_proba(row)[0, 1]
st.metric("Churn risk", f"{p:.0%}")
push = pd.Series(model.coef_[0] * z, index=features)
st.bar_chart(push, horizontal=True)
```

The whole app is [app.py](app.py), about 60 lines. **Run it** from this folder (Windows, macOS and Linux):

```bash
python -m streamlit run app.py
```

Your browser opens at http://localhost:8501. Change a number and the page updates. Stop it with Ctrl+C in the terminal.

## 7. On real data

The notebook computes the contributions above from the bundle, draws the same bar chart the app shows, then **tests the app with AppTest**:

- With the default inputs (the 05.10 customer) the app shows **52%** and recommends a voucher.
- Entering the loyal customer drops the risk to **8%** and the advice to "no voucher", matching the hand calculation.
- Raising the voucher cost to £35 moves the break-even threshold to 58% and the 52% customer no longer gets one. (At £30 the threshold is 50%, still below 52%: the advice would not change.)
- Entering 200 days since the last order with 150 days of tenure shows an error instead of a score.

## 8. Where it breaks

- **Forgetting the cache.** Without `@st.cache_resource` the model is reloaded on every click; with a big model the page crawls.
- **Explanations are not causes.** "Few orders raises the risk" describes the model, not the customer. Sending more emails to raise the order count will not necessarily keep anyone.
- **Contributions depend on the scaling.** They are relative to an *average* customer in the training data. Explain them that way.
- **Only for linear models.** For a tree ensemble use SHAP values (05.8) instead of $w z$; the idea, contributions that add up to the score, is the same.
- **A demo is not a product.** Streamlit apps have no logins by default and are single-file scripts. Fine for an internal tool; not for customers.
- **Inputs a human can mistype.** Validate (recency ≤ tenure) and stop with a clear message, as the app does.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: the contributions and probability for 1 order, £80, 3 products, 273 days tenure, 273 days since last order (the riskiest customer from 05.8). Which feature dominates?
2. Add a second tab to a copy of the app that accepts a CSV upload (`st.file_uploader`) and shows a table of scores. Test it with AppTest or by running it.
3. Change the app so the threshold can be "break-even" or "saved bundle threshold", chosen with a radio button. Find a customer where the two disagree.

**Quiz**

1. What happens in a Streamlit app when someone moves a slider?
2. Why does loading the model need `@st.cache_resource`?
3. What do the bars in "What drives this score" add up to?
4. At £8 voucher and £60 lost margin, what is the break-even threshold?
5. Why might the app and the bundle disagree about a voucher at 7.5% risk?

## 10. Recap

- Streamlit turns a script into a web page; every interaction **reruns the script**, so **cache** the model.
- Logistic regression explains itself: contributions $w_j z_j$ add up to the log-odds.
- Let the business change the costs; the **break-even threshold** is $c_\text{voucher}/c_\text{lost}$.
- Test apps with **AppTest**; run them with `python -m streamlit run app.py`.
- Present contributions as "what the model weighs", never as causes.

**Next:** [10.4 Docker in one hour](../04-docker/) (optional) or [10.5 Experiment tracking with MLflow](../05-mlflow/)
