# 10.5 · Experiment tracking with MLflow

⏱ about 1.5 hours · Prerequisites: 05.10, 10.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Over Modules 03 to 05 you trained dozens of churn models. Quick: which settings gave the random forest its 0.736? Which notebook made the model now in `models/`? Was the threshold 0.07 chosen before or after the switch to five features?

Nobody remembers, and notebook outputs get overwritten every time you re-run them. When the CEO asks "why this model and not the one from August?", you need a record, not a memory.

## 2. The idea

A **lab notebook for models**. Every time you train something, write down automatically what you tried, how well it did, and keep the trained model. Later you can sort, compare, reload and rebuild any of them.

MLflow does this with a few lines of code. It stores runs in a small database and files on disk (here `mlruns/` at the course root), and gives you a web page to browse them.

## 3. Definition

- **Experiment**: a named group of related runs (`shopwave-churn`).
- **Run**: one training attempt. It records:
  - **parameters**: inputs you chose (`max_depth=3`, `split_seed=42`);
  - **metrics**: numbers you measured (`cv_auc=0.740`);
  - **artifacts**: files (the trained model, plots);
  - **tags**: labels (`purpose=lesson 10.5`).
- **Logged model**: a model saved with its environment and an input example, so it can be reloaded anywhere.
- **Model registry**: named models with numbered **versions**. An **alias** (`@champion`) points at the version in use and can be moved.
- **Tracking URI**: where runs are stored. Here a SQLite file, `sqlite:///mlruns/mlflow.db`.

## 4. Formula

Nothing new to compute; one rule to follow. A run is reproducible when it records

$$\text{run} = (\text{code version},\ \text{data version},\ \text{parameters incl. seeds},\ \text{library versions}) \;\to\; (\text{metrics},\ \text{model})$$

and choosing between runs uses the **one-standard-error rule** from 05.10:

$$\text{bar} = \text{cv\_auc}_\text{best} - \text{std}_\text{best}, \qquad \text{tied} = \{m : \text{cv\_auc}_m \ge \text{bar}\}$$

## 5. By hand

The four runs the notebook logs (5-fold CV on 2,487 training customers; test on 830):

| run | CV AUC | CV std | test AUC | test cost (£, threshold 0.07) |
|---|---|---|---|---|
| logistic_regression | 0.740 | 0.029 | 0.741 | 3,945 |
| random_forest | 0.736 | 0.030 | 0.731 | 3,780 |
| tree_depth_3 | 0.731 | 0.024 | 0.724 | 4,150 |
| tree_depth_8 | 0.683 | 0.020 | 0.649 | 5,600 |

Bar = 0.740 − 0.029 = **0.711**. Tied: logistic regression, random forest, depth-3 tree. The depth-8 tree is clearly worse: it memorises the training customers.

Among the tied models, logistic regression is the simplest, so it becomes the champion. Notice the random forest has the **lowest test cost**: a different metric would crown a different winner. Choose the metric that matches the decision **before** looking at results, and use the test set only to confirm.

## 6. In code

```python
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("shopwave-churn")

with mlflow.start_run(run_name="random_forest"):
    mlflow.log_params({"n_estimators": 300, "min_samples_leaf": 10, "split_seed": 42})
    mlflow.log_metrics({"cv_auc": 0.736, "test_auc": 0.731})
    info = mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                    skops_trusted_types=["sklearn.tree._tree.Tree"])

mlflow.search_runs(experiment_names=["shopwave-churn"], order_by=["metrics.cv_auc DESC"])
mlflow.register_model(info.model_uri, "shopwave-churn")
mlflow.sklearn.load_model("models:/shopwave-churn@champion")
```

**The web UI**, from the course root (Windows, macOS and Linux):

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

then open http://127.0.0.1:5000. Ctrl+C stops it.

## 7. On real data

The notebook:

- logs the four runs above in about **12 seconds**;
- applies the one-standard-error rule to the run table;
- registers logistic regression as version 1 of `shopwave-churn` with alias `@champion`, reloads it by name and confirms it predicts **exactly** like the bundle from 10.1;
- rebuilds the random forest from nothing but its logged parameters and gets the **same** test AUC (0.7307), because the seeds were logged too.

## 8. Where it breaks

- **Logging only the winners.** The losers are evidence too ("we tried deeper trees; they overfit"). Log everything.
- **Not logging seeds and data versions.** Then "rebuild run X" is impossible. Log the split seed, the data cut-off date and, in a real project, the git commit.
- **Comparing by test score.** Choosing on the test set turns it into a training set. Compare on CV metrics; open the test set once.
- **Loading untrusted model files.** MLflow 3 saves scikit-learn models with **skops**, which refuses unknown types instead of executing them. Tree models need `skops_trusted_types=["sklearn.tree._tree.Tree"]`; only add that for models you trained yourself.
- **A local database is personal.** `mlruns/mlflow.db` lives on your laptop. Teams run one shared MLflow server (or a managed one) so everyone logs to the same place.
- **Clutter.** A hundred runs named `run_1` are as useless as none. Name runs and tag their purpose.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Log a fifth model (KNN, k = 51) and find its place on the leaderboard.
2. Log the logistic regression's ROC curve as an image artifact.
3. Register the random forest as `@challenger`; compare it with `@champion`. What evidence would you need to promote it?

**Quiz**

1. Name the four kinds of things an MLflow run records.
2. Why sort runs by CV AUC rather than test AUC?
3. What does an alias like `@champion` give you that a run ID does not?
4. Besides parameters, what must a run record to be rebuilt later?
5. Why does MLflow's skops format refuse some types, and why is that safer than a pickle?

## 10. Recap

- Track every training run: **parameters, metrics, artifacts, tags**.
- Compare runs in one table; choose with the **one-standard-error rule** on CV metrics, confirm once on test.
- Register the chosen model; point **`@champion`** at it; load it by name.
- Log **seeds and data versions** so any run can be rebuilt exactly.
- `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` to browse.

**Next:** [10.6 Monitoring, data drift, concept drift, retraining](../06-monitoring-drift/)
