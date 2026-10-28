# 05.10 · Choosing and saving the final model

⏱ about 0.5 hours · Prerequisites: 03.7 (cost-based thresholds), 05.6, 05.8 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You have five churn models. They all work. The CEO wants one in production on Monday, and the engineering team asks two practical questions: *"Which one?"* and *"What file do we deploy, and how do we know it's the right one in six months?"*

Picking the top of the leaderboard feels obvious. It often isn't right. And a model saved as `model.pkl` with no record of its features, threshold or performance is a future incident waiting to happen.

## 2. The idea

Choosing a model is a business decision informed by evidence, not a sort by score:

- **Is the difference real?** If two scores differ by less than the noise, they are tied.
- **Among ties, prefer simple.** Simpler models are faster, cheaper, easier to explain, and usually more robust when data drifts.
- **Then fit the business.** Speed, explainability, cost of errors, regulation.

Saving a model means saving everything needed to use it correctly later: the fitted pipeline, the feature list, the decision threshold, the metrics, the training data description, the date and library versions. Then a one-page **model card** for humans.

## 3. Definition

- **Leaderboard**: a table comparing candidate models on the same CV folds, with score, stability and cost columns.
- **One-standard-error rule**: choose the simplest model whose CV score is within one standard deviation of the best.
- **Model bundle**: a single saved object containing the pipeline and its metadata.
- **Versioned file name**: a name that encodes model type, date and key metric, e.g. `churn_logreg_2026-10-28_auc0.741.joblib`.
- **Model card**: a short document describing a model's purpose, data, performance, decision rule and limitations.

## 4. Formula

**One-standard-error rule**:

$$\text{bar} = \text{score}_{\text{best}} - \text{std}_{\text{best}} \qquad \text{tied} = \{\,m : \text{score}_m \ge \text{bar}\,\}$$

then choose the simplest model in the tied set.

**Break-even threshold** for a voucher decision (from 03.7's cost model): send a voucher when the expected loss avoided exceeds its cost:

$$p \cdot c_{\text{lost}} > c_{\text{voucher}} \quad\Longrightarrow\quad t^* = \frac{c_{\text{voucher}}}{c_{\text{lost}}}$$

This assumes the model's probabilities are calibrated; in practice you confirm it on cross-validated predictions.

## 5. By hand

**a) The leaderboard** (5-fold CV on 2,487 training customers):

| model | CV AUC | CV std |
|---|---|---|
| LightGBM (tuned) | 0.744 | 0.030 |
| random forest | 0.743 | 0.031 |
| logistic regression | 0.740 | 0.029 |
| KNN (k = 51) | 0.732 | 0.033 |
| decision tree (depth 3) | 0.731 | 0.024 |

Bar = 0.744 − 0.030 = **0.714**. Every model is at or above 0.714, so **all five are tied**. Simplest candidates: the depth-3 tree and logistic regression. Logistic regression has the higher score, smooth probabilities (the tree has only 8 leaves, so only 8 possible probabilities) and readable coefficients. **Choose logistic regression.**

**b) The threshold.** £5 voucher, £60 lost customer:

$$t^* = \frac{5}{60} = 0.083$$

Any customer with more than an 8.3% chance of churning is worth a voucher. Cross-validation on training data picked **0.07**, close to the theory.

## 6. In code

```python
r = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc", return_train_score=True)
r["test_score"].mean(), r["test_score"].std(), r["fit_time"].mean()

bundle = {"pipeline": final, "features": features, "threshold": 0.07,
          "metrics": {"cv_auc": 0.740, "test_auc": 0.741},
          "trained": "2026-10-28", "sklearn_version": sklearn.__version__}
joblib.dump(bundle, "models/churn_logreg_2026-10-28_auc0.741.joblib")

loaded = joblib.load(path)
row = pd.DataFrame([new_customer])[loaded["features"]]
p = loaded["pipeline"].predict_proba(row)[0, 1]
```

Save the **whole pipeline** (scaler and model together), never the model alone. Otherwise someone will forget to scale the inputs in production.

## 7. On real data

- Leaderboard as above. The random forest's train AUC (0.80) is well above its CV AUC (0.74): it memorises a little. Logistic regression's train and CV scores are almost identical (0.744 vs 0.740): what it learned generalises.
- Chosen model: logistic regression. Test AUC (opened once): **0.741**.
- Threshold 0.07 flags about **94%** of test customers. Cost £3,945 against £4,150 for "voucher to everyone" and £21,720 for "voucher to nobody". At a £5 voucher the model's job is mostly to skip the customers who clearly need nothing; at a £20 voucher the threshold would rise to about 0.33 and the model would matter much more.
- The bundle is about 2 KB. A new customer (2 orders, £310, 14 products, 150 days tenure, 95 days since last order) scores **0.52** → voucher.
- The model card is generated from the bundle, so its numbers can never drift from the file.

## 8. Where it breaks

- **Sorting by score alone.** Picks noise winners and complex models whose edge disappears in production.
- **Saving the model without its preprocessing.** Production feeds unscaled data and gets nonsense, silently.
- **Saving without the feature order.** A DataFrame with the same columns in a different order can be scored wrongly by some libraries. Always select `df[bundle["features"]]`.
- **Library version drift.** A pickle saved with one scikit-learn version may not load, or may behave differently, in another. Record the version; check it on load.
- **joblib/pickle files are code.** Loading one can run arbitrary code. Only load files you or your team created.
- **Overwriting `model.joblib`.** Nobody can tell which version is live. Put date and metric in the name, and keep old versions.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: apply the one-standard-error rule to a four-model leaderboard, and choose for two different business situations.
2. Save the tuned LightGBM as a bundle and compare its decision for the same new customer.
3. Write a `score_customers(path, df)` function that checks for missing features.

**Quiz**

1. Why is "highest CV score" not enough to choose a model?
2. Best CV AUC 0.750 with std 0.020. Which of 0.745, 0.735 and 0.720 are tied with it?
3. Name five things a saved model bundle should contain.
4. Why put the date and metric in the file name?
5. What sections does a model card have?

## 10. Recap

- Build a **leaderboard** on identical CV folds: score, std, train-vs-CV gap, speed, interpretability.
- Differences within one std are **ties**; among ties, **choose the simplest**.
- The decision threshold comes from costs: $t^* = c_{\text{voucher}} / c_{\text{lost}}$, confirmed on CV predictions.
- Save a **bundle**: pipeline + features + threshold + metrics + data description + date + versions, in a **versioned file**.
- Write a **model card** so humans know what the model is for and where it must not be used.

**Next:** [05.P Mini-project: a tuned, explained churn model](../project/)
