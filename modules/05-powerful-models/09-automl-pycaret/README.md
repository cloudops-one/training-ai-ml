# 05.9 · AutoML in industry: PyCaret, and what it automates

⏱ about 1 hour · **Optional** · Prerequisites: 04.8 (pipelines), 05.6, 05.8, 05.10 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A colleague from the analytics team looks at your churn notebooks and says: *"You spent two weeks on this? I'd have had a leaderboard in ten minutes with PyCaret."*

They are partly right. Many companies use **AutoML** tools for quick baselines and prototypes, and you will meet them at work. They are also partly wrong: those ten minutes would not have caught the leaky `recency_days` column (01.9), questioned accuracy on a 5% target (05.7), or worked out that a £5 voucher should go to 94% of customers (05.10).

This lesson shows what AutoML does, by mapping each PyCaret call onto something you have already built by hand. The goal is not to replace the basics. It is to recognise AutoML as a shortcut through steps you already understand.

## 2. The idea

AutoML tools automate the repetitive parts of a modelling project:

```
your manual work (Modules 04–05)            PyCaret
────────────────────────────────            ───────────────────
Pipeline: impute, scale, encode   ───────►  setup()
CV leaderboard of many models     ───────►  compare_models()
RandomizedSearchCV / Optuna       ───────►  tune_model()
SHAP beeswarm and waterfall       ───────►  interpret_model()
refit on all training data, joblib ──────►  finalize_model(), save_model()
```

Underneath, they call the same libraries you used: scikit-learn, LightGBM, SHAP. Same algorithms, same data, same pitfalls. What changes is how much you type, and how much is hidden from you.

## 3. Definition

- **AutoML**: software that automates parts of the ML workflow, typically preprocessing, model selection, hyperparameter tuning and sometimes feature engineering or ensembling.
- **Low-code ML**: AutoML exposed through a few high-level function calls (PyCaret's style).
- **PyCaret**: an open-source, low-code Python library that wraps scikit-learn, LightGBM, XGBoost, SHAP and others behind functions like `setup` and `compare_models`.

Other tools you will hear about:

| Tool | Type | Typical use |
|---|---|---|
| **PyCaret** | open source, low-code wrapper | fast baselines in a notebook |
| **FLAML** (Microsoft) | open source, cost-aware tuning | good models on a tight compute budget |
| **AutoGluon** (Amazon) | open source, stacking ensembles | strong accuracy with little effort, heavier compute |
| **H2O AutoML** | open source, Java backend | large tabular data, enterprise deployments |
| **Cloud AutoML** (Vertex AI, SageMaker Autopilot, Azure AutoML) | managed services | teams without ML engineers; pay per training hour |

## 4. Formula

There is no new maths here; every formula is one you already know. The useful "formula" is the cost of the search AutoML runs on your behalf:

$$\text{fits} = (\text{models compared} \times k) + (\text{tuning iterations} \times k) + \text{refits}$$

With 7 models, 5 folds and 20 tuning iterations: $7 \times 5 + 20 \times 5 + 2 = 137$ fits behind three lines of code. That is why AutoML feels like magic on small data and gets slow on large data.

## 5. By hand

Before running anything, fill in the right-hand column from memory. Each PyCaret call does something you have already done:

| PyCaret call | What it does | Where you did it by hand |
|---|---|---|
| `setup(data, target, normalize=True, fold=5)` | | |
| `compare_models(sort="AUC")` | | |
| `tune_model(model, n_iter=20)` | | |
| `interpret_model(model)` | | |
| `finalize_model(model)` | | |
| `save_model(model, name)` | | |

Answers: imputer + scaler pipeline and fixed stratified 5-fold CV (04.7–04.8); cross-validated leaderboard (05.10); random search (05.6); SHAP beeswarm (05.8); refit on all training data (05.10); joblib dump of the whole pipeline (05.10).

Then count: with 7 models, 5 folds and 20 tuning iterations, how many fits? (137, from section 4.)

## 6. In code

```python
from pycaret.classification import setup, compare_models, create_model, tune_model, \
                                    interpret_model, finalize_model, save_model, pull

setup(data=X_train.assign(churned=y_train), target="churned",
      session_id=42, fold=5, normalize=True, verbose=False)
best = compare_models(include=["lr", "dt", "knn", "rf", "et", "gbc", "lightgbm"], sort="AUC")
tuned = tune_model(create_model("lightgbm"), optimize="AUC", n_iter=20)
interpret_model(tuned)
save_model(finalize_model(tuned), "models/churn_pycaret_lightgbm")
```

The notebook then repeats every step in plain scikit-learn in about 30 lines, with each line labelled by the PyCaret call it replaces.

**Environment.** PyCaret 3.3.2 (the current release) runs only on Python 3.9–3.11 and pins scikit-learn 1.4, pandas 2.1 and NumPy 1.26. The course uses Python 3.12, and so does Google Colab, so PyCaret needs its own environment: see [setup/requirements-pycaret.txt](../../../setup/requirements-pycaret.txt). In the main course environment the notebook detects this and skips the PyCaret cells; the scikit-learn section runs everywhere.

## 7. On real data

Run in a Python 3.11 environment, PyCaret's leaderboard on the churn training set (5-fold CV) was:

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.760 | 0.701 | 0.668 |
| Gradient Boosting | 0.735 | 0.673 | 0.629 |
| Random Forest | 0.722 | 0.658 | 0.595 |
| Extra Trees | 0.710 | 0.651 | 0.575 |
| LightGBM | 0.707 | 0.647 | 0.590 |
| K Neighbors | 0.703 | 0.654 | 0.603 |
| Decision Tree | 0.603 | 0.610 | 0.550 |

Tuned LightGBM: CV AUC 0.755, test AUC **0.739**.

The plain scikit-learn version in the notebook gives the same story with slightly different numbers (different random folds): logistic regression first (0.740), unlimited decision tree last (0.594), tuned LightGBM CV 0.742 and test 0.734.

Both agree with everything Module 05 found by hand: on this table, the powerful models tie or trail logistic regression, and tuning mostly repairs bad defaults. AutoML did not find anything you missed, because there was nothing to find.

## 8. Where it breaks

- **Leakage passes straight through.** In exercise 3 a leaky column gives every model AUC **1.000** in PyCaret, with no warning. AutoML optimises whatever you give it.
- **Hidden defaults.** Which imputation, which CV scheme, which search space? You can find out, but you have to go looking. In production you must know.
- **Dependency pins.** PyCaret pins old library versions and lags behind Python releases. Mixing it into a production environment can break other code.
- **Wrong metric, wrong threshold.** AutoML sorts by a metric you name. It does not know that a missed churner costs £60 and a voucher £5.
- **Explaining "why this model".** "PyCaret picked it" is not an answer a model card or an auditor will accept.
- **Compute cost at scale.** 137 fits on 2,500 rows takes seconds; on 50 million rows it is a bill.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On paper: write the scikit-learn equivalent of each PyCaret call.
2. Write your own five-line "AutoML" leaderboard in scikit-learn.
3. (PyCaret environment only) Add a leaky column and see whether PyCaret warns you.

**Quiz**

1. Map each of PyCaret's `setup`, `compare_models`, `tune_model`, `interpret_model`, `finalize_model` to the lesson where you did the same by hand.
2. When is AutoML the right tool?
3. Give three risks of relying on AutoML for a production model.
4. Why do PyCaret and your hand-built leaderboard reach the same conclusion on the churn data?
5. Why does PyCaret need a separate environment in this course?

## 10. Recap

- AutoML tools such as **PyCaret** wrap the steps you already know: pipeline, CV leaderboard, tuning, SHAP, refit and save.
- They are widely used in industry for **fast baselines and prototypes**, not as a substitute for understanding.
- Underneath they use the **same libraries**, so they reach the same conclusions and fall into the same traps, including leakage.
- PyCaret 3.3 needs **Python 3.9–3.11** in its own environment.
- AutoML automates the typing, not the thinking.

**Next:** [05.10 Choosing and saving the final model](../10-choosing-and-saving/)
