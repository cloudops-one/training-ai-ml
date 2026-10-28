# Module 10 · Shipping & Running ML

**ShopWave asks:** *"The churn model lives in your notebook. Retention needs scores every Monday, the website wants a live 'risk' badge, and nobody will notice if it quietly breaks."*

**Time:** about 10 hours. **Prerequisites:** Module 05 (the chosen churn model, 05.10), Module 03.7 (voucher costs), Module 09.10 (the RAG chatbot, for the project).

## Where you are

You have a good churn model: scaled logistic regression, test AUC 0.741, a voucher threshold chosen from costs. Right now it is a notebook cell. Nobody else can use it, nobody can tell which version produced last month's scores, and if customer behaviour changes nobody will know until the campaign fails.

This module turns the notebook into something the business can rely on:

- a **saved, versioned model** and a script that scores a CSV every Monday (10.1),
- a **web API** the website can call (10.2) and a **web app** the retention team can click (10.3),
- a **container** that runs the same way on any machine (10.4),
- a **record of every experiment** so you can say which model is live and why (10.5),
- **monitoring** that notices when the world drifts away from the training data (10.6),
- and a check that the model is **fair, private and governable** (10.7).

None of this needs new maths. It needs care, a few new tools, and the habit of asking "what happens when this runs without me?"

## What you will be able to do afterwards

- Save a full pipeline with its threshold, metrics and versions; reload it safely; reproduce it.
- Write a batch scoring script and schedule it on Windows, macOS or Linux.
- Serve a model with FastAPI, validate inputs, test it without starting a server, and call it from curl or PowerShell.
- Build a Streamlit app that explains each prediction in business terms.
- Package a service in a Docker image with pinned dependencies.
- Track experiments and register a "champion" model with MLflow.
- Monitor data drift (PSI, KS test), calibration and AUC over time, and choose a retraining policy from evidence.
- Measure group fairness, protect personal data, and write a model card a reviewer can sign off.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 10.1 | [Saving and loading models; reproducibility](01-saving-and-reproducibility/) | 0.5 h |
| 10.2 | [A prediction API with FastAPI](02-fastapi/) | 1.5 h |
| 10.3 | [A web app with Streamlit](03-streamlit/) | 1 h |
| 10.4 | [Docker in one hour](04-docker/) *(optional)* | 1 h |
| 10.5 | [Experiment tracking with MLflow](05-mlflow/) | 1.5 h |
| 10.6 | [Monitoring, data drift, concept drift, retraining](06-monitoring-drift/) | 1.5 h |
| 10.7 | [Responsible AI: fairness, privacy, explainability, governance](07-responsible-ai/) | 1 h |
| 10.P | [Mini-project: deploy the churn model and the chatbot](project/) | 2 h |

## Files you will run

Unlike earlier modules, several lessons ship **programs**, not just notebooks. They live next to each lesson and all share one file, [`lib/churn_model.py`](../../lib/churn_model.py), so the training script, the API and the app can never disagree about features or thresholds.

| File | What it does | Start it with |
|---|---|---|
| `01-saving-and-reproducibility/train_churn.py` | trains the model, saves `models/churn_bundle.joblib` | `python train_churn.py` |
| `01-saving-and-reproducibility/score_customers.py` | scores a CSV of customers | `python score_customers.py customers.csv` |
| `02-fastapi/api.py` | the churn API | `python -m uvicorn api:app --reload` |
| `03-streamlit/app.py` | the churn checker web app | `python -m streamlit run app.py` |
| `04-docker/Dockerfile` | the API in a container | `docker build -f modules/10-deployment-mlops/04-docker/Dockerfile -t shopwave-churn .` |
| `project/` | API with logging, two-tab app, tracked training | see the project README |

Run each command from the folder it lives in (the Docker build from the course root), with the course environment active. The commands are the same on Windows, macOS and Linux; where a command differs, the lesson shows both. Generated files (models, experiment logs, prediction logs) go to `models/` and `mlruns/` at the course root, which git ignores.

## What the module finds

- The saved bundle is 2 KB and reproduces its own predictions exactly; you can recompute one by hand from the numbers inside it (10.1).
- One prediction through the API takes a couple of milliseconds; most of a real request's time is the network (10.2).
- Across twelve months of real data, **input drift alarms fire every winter while the model's ranking (AUC) holds**, but its **calibration** does not: in January it predicts 40% churn when 65% happen. Retraining on the latest months makes September *worse*; the same-season model is best (10.6).
- UK and non-UK customers get almost the same recall and false-positive rate, and the gap is far smaller than the uncertainty from a group of only 81 test customers (10.7).

## The maths in this module

Percentiles for latency (10.2), the logistic regression formula from 03.4 taken apart into per-feature contributions (10.3), the Population Stability Index and the Kolmogorov–Smirnov distance (10.6), and rates and differences for fairness (10.7). Each is computed by hand on a few numbers first.
