# 10.1 · Saving and loading models; reproducibility

⏱ about 0.5 hours · Prerequisites: 05.10 (choosing and saving) · Notebook: [lesson.ipynb](lesson.ipynb) · Programs: [train_churn.py](train_churn.py), [score_customers.py](score_customers.py)

---

## 1. The problem

Retention wants churn scores for every customer **every Monday at 7am**, as a CSV they can open in Excel. You will not be at your laptop at 7am, and in six months somebody will ask: *"Which model produced the scores we sent in March, and can you produce them again?"*

A notebook cannot answer either request. It needs a person to press "Run all", and its results depend on whatever happened to be in memory and installed that day.

## 2. The idea

Split the work into two programs that never need a notebook:

1. **Train once**, save everything the model needs into one file (the *bundle*).
2. **Score many times** with a small script that loads the bundle and reads a CSV.

Then make the training repeatable: same data, same code, same library versions, same random seed → same model. Write those four things down and anyone can rebuild it.

## 3. Definition

- **Serialisation**: turning a Python object (the fitted pipeline) into bytes in a file, and back. `joblib.dump` / `joblib.load`.
- **Bundle**: a dictionary saved as one file: the fitted pipeline, the feature list, the decision threshold, the metrics, a data description, the training date and the scikit-learn version.
- **Batch scoring**: scoring many rows at once on a schedule (nightly, weekly), as opposed to *online* scoring one request at a time (10.2).
- **Reproducibility**: someone else, later, can rebuild the same model and get the same numbers.
- **Pinning**: writing exact library versions (`scikit-learn==1.9.1`) instead of ranges, so a future install cannot silently change behaviour.

## 4. Formula

There is no new maths. There is a recipe:

$$\text{model} = f(\text{data},\ \text{code},\ \text{library versions},\ \text{random seed})$$

Change any one of the four and you may get a different model. The saved bundle lets you **use** the model without rebuilding it; the four ingredients let you **rebuild** it.

And the prediction the bundle makes is the logistic regression formula from 03.4, with the scaler from 04.5 in front:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}, \qquad p = \sigma\Big(b_0 + \sum_j w_j z_j\Big)$$

## 5. By hand

A saved model is not magic: it is the numbers above. Here they are, read out of `churn_bundle.joblib`, for the customer from 05.10 (2 orders, £310, 14 products, 150 days tenure, 95 days since last order):

| feature | value $x$ | mean $\mu$ | std $\sigma$ | $z$ | weight $w$ | $w \cdot z$ |
|---|---|---|---|---|---|---|
| orders | 2 | 3.37 | 5.19 | −0.264 | −0.992 | +0.262 |
| spend | 310 | 1,510.12 | 5,346.76 | −0.224 | +0.060 | −0.013 |
| products | 14 | 48.45 | 61.67 | −0.559 | −0.450 | +0.252 |
| tenure_days | 150 | 176.08 | 78.01 | −0.334 | −0.173 | +0.058 |
| days_since_last | 95 | 92.22 | 77.41 | +0.036 | +0.437 | +0.016 |

$b_0 = -0.476$, so $b_0 + \sum w z = -0.476 + 0.262 - 0.013 + 0.252 + 0.058 + 0.016 = 0.099$ (0.098 with the unrounded numbers), and

$$p = \frac{1}{1 + e^{-0.098}} = 0.524$$

The bundle's threshold is 0.07, so this customer gets a voucher. The whole model is **16 numbers** plus a threshold: five means, five standard deviations, five weights and an intercept. The file is about 2 KB.

## 6. In code

```python
import joblib, sklearn
bundle = {"pipeline": pipeline, "features": FEATURES, "threshold": 0.07,
          "metrics": {"cv_auc": 0.738, "test_auc": 0.741},
          "trained": "2026-10-28", "sklearn_version": sklearn.__version__}
joblib.dump(bundle, "models/churn_bundle.joblib")

bundle = joblib.load("models/churn_bundle.joblib")
p = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
```

The course packages this in [`lib/churn_model.py`](../../../lib/churn_model.py) so every program shares it, and ships two scripts:

```bash
python train_churn.py                      # writes models/churn_bundle.joblib
python score_customers.py customers.csv    # writes customers_scored.csv
```

## 7. On real data

The notebook trains the bundle (under a second from the cached data), reloads it and confirms the predictions are **identical**, recomputes the table above from the file, and shows:

- **Seeds matter.** The same code with five different `random_state` values for the train/test split gives test AUCs from **0.730 to 0.744**. Record the seed, or nobody can reproduce "0.741".
- **Versions are recorded.** The bundle stores scikit-learn's version, and loading it under a different version prints a warning instead of failing silently.
- **Batch scoring.** 300 customers written to a CSV, two of them deliberately broken. The script skips and reports the broken rows, scores the rest, and writes a file retention can open.

## 8. Where it breaks

- **Saving the model without the scaler.** Production feeds raw pounds into a model that expects z-scores. Always save the whole pipeline.
- **Library upgrades.** A pickle from scikit-learn 1.9 may not load in 1.11, or may load and behave differently. Pin versions (`pip freeze > requirements-lock.txt`) and check the stored version on load.
- **Loading files you did not make.** `joblib.load` can run arbitrary code hidden in the file. Only load bundles you or your team created. (MLflow's safer format appears in 10.5.)
- **Silent bad input.** A script that fills missing values with 0 produces confident nonsense. Reject and report bad rows instead.
- **"It works on my machine."** Paths like `C:\Users\me\...` or `/Users/me/...` break on every other machine. The scripts find the course folder themselves.

### Scheduling the Monday run

| System | How |
|---|---|
| **macOS / Linux** | `crontab -e`, then add `0 7 * * 1 cd /path/to/course/modules/10-deployment-mlops/01-saving-and-reproducibility && /path/to/course/.venv/bin/python score_customers.py /data/customers.csv` |
| **Windows** | Task Scheduler → Create Basic Task → Weekly, Monday 07:00 → Start a program: `C:\path\to\course\.venv\Scripts\python.exe`, arguments `score_customers.py C:\data\customers.csv`, start in the script's folder. Or in PowerShell: `schtasks /Create /SC WEEKLY /D MON /ST 07:00 /TN ShopWaveScores /TR "C:\path\to\course\.venv\Scripts\python.exe C:\path\to\...\score_customers.py C:\data\customers.csv"` |

Use the full path to the environment's Python: the scheduler does not "activate" anything for you.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: recompute the probability for a customer with 10 orders, £2,000, 60 products, 300 days tenure, 20 days since last order. Check with the bundle.
2. Add a `--threshold` option to a copy of `score_customers.py` so retention can try a stricter policy without retraining.
3. Write a function `check_bundle(path)` that loads a bundle and reports: missing keys, version mismatch, and whether the model still scores AUC above 0.70 on a given labelled table.

**Quiz**

1. Name the four ingredients that decide which model you get.
2. Why save the whole pipeline rather than just the logistic regression?
3. What is the difference between batch and online scoring? Give a ShopWave example of each.
4. Why is `joblib.load` on a file from the internet dangerous?
5. Your scheduled task runs but produces an import error. What is the most likely cause?

## 10. Recap

- Save a **bundle**: pipeline + features + threshold + metrics + data description + date + versions.
- A saved logistic regression is a handful of numbers; you can recompute any prediction by hand.
- **Same data, code, versions and seed** → same model. Record all four.
- Batch scoring is a small script plus a scheduler (cron on macOS/Linux, Task Scheduler on Windows), using the environment's full Python path.
- Validate inputs and report bad rows; never guess.

**Next:** [10.2 A prediction API with FastAPI](../02-fastapi/)
