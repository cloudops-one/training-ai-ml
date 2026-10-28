# 10.P · Mini-project: deploy the churn model and the chatbot

⏱ about 2 hours · Prerequisites: all of Module 10, 09.10 (RAG) · Notebooks: [project.ipynb](project.ipynb) (your work), [solution.ipynb](solution.ipynb) (check afterwards)

---

## The brief

ShopWave is ready to put two things in front of staff:

- the **churn model**, as an API for the website and a page for the retention team;
- the **support chatbot** from Module 09, answering policy questions from ShopWave's own documents.

Before launch, the head of operations wants evidence that each piece works, a first monitoring report, and a release note people can follow. The programs are written and in this folder. Your job is to run them, test them, monitor them and report, the way a team would before switching something on.

| File | What it is |
|---|---|
| [train.py](train.py) | trains the model, saves `models/churn_bundle.joblib`, records the run in MLflow (the bundle stores the run ID) |
| [api.py](api.py) | the 10.2 API plus `/model` (purpose and metrics) and a **prediction log**: every prediction is appended to `models/prediction_log.csv` |
| [support_bot.py](support_bot.py) | retrieval over `data/shopwave_docs/` with local embeddings, answers via `lib/course_llm.py` |
| [app.py](app.py) | a Streamlit app with two tabs: **Churn risk** and **Support chatbot**; the chatbot tab explains what to do if no LLM is configured |

## Tasks

1. **Train and record.** Run `train.py`; find the run in the MLflow UI; check the bundle knows its run ID.
2. **Test the API.** `/health`, `/model`, a good `/predict`, two kinds of bad input.
3. **A live month.** Score every customer as of 1 December 2011 (274-day history, as in training) through `/predict/batch`, so the log fills up.
4. **Monitoring report.** PSI, medians and a status for each feature and for the score, from the log against the training customers. Three sentences: what changed, what it means, what you recommend. Use what 10.6 found.
5. **The chatbot.** At least five questions: three answered directly by the documents, one answered only indirectly, one not covered. Record answers, sources and timing. Explain any failure.
6. **Smoke-test the app** with `AppTest`, with and without an LLM configured.
7. **Release note** for the retention and support teams (half a page).

## Running it

Run each command from **this folder**, with the course environment active. They are the same on Windows, macOS and Linux.

```bash
python train.py                           # once, and after any change to the model
python -m uvicorn api:app                 # the API: http://127.0.0.1:8000/docs
python -m streamlit run app.py            # the app: http://localhost:8501
```

Run the API and the app in separate terminals; stop each with Ctrl+C. Browse experiment runs from the **course root** with `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

The chatbot tab needs an LLM: the free option is Ollama with `ollama pull llama3.2`, or a key in `.env` (see [setup/README.md](../../../setup/README.md), including which model suits your computer's memory). The churn tab and the API work without one.

Activating the environment: `source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\Activate.ps1` in Windows PowerShell (from the course root).

## Rules

- Every number in your report comes from an executed cell.
- Say what you could not check (for example, AUC for December: its labels arrive on 1 March 2012).
- An alarm is a question, not an answer: investigate before you recommend action.

## Rubric (20 points)

| Area | Points | Full marks when… |
|---|---|---|
| Training and tracking | 3 | Run recorded with parameters, metrics and the bundle; bundle and run point at each other |
| API checks | 3 | Health, model card, one good and two bad requests, with status codes explained |
| Monitoring report | 5 | PSI and medians for every feature and the score; each alarm explained (including artefacts); recommendation uses 10.6's calibration finding; label delay stated |
| Chatbot evaluation | 4 | Five questions of the three kinds; sources checked against the documents; failures explained honestly |
| App smoke test | 2 | No exceptions; churn tab and chatbot work; graceful message without an LLM |
| Release note | 3 | How to start it on each OS; purpose and out-of-scope uses; monitoring plan; known limits |

## Stretch

- Fix the tenure artefact: compute `tenure_days` from all history, retrain, and check whether the "act" alarm disappears.
- Add a `/feedback` endpoint to the API that records whether a flagged customer actually ordered, and turn it into the label source for monitoring.
- Build the project's Docker image (10.4) with the prediction log written to a mounted folder (`docker run -v ...`).

## What the solution found

So you can check your direction:

- The production bundle reproduces 10.1 (test AUC 0.741, threshold 0.07) and stores its MLflow run ID.
- 3,956 customers scored for 1 December 2011; vouchers for 92%. Tenure PSI 0.27 (**act**, but the 274-day window artefact from 10.6), recency 0.15 (**watch**: Christmas shopping, median 41 days against 72), score PSI 0.07 (**ok**).
- The real risk is calibration: the model predicts 40% churn, and last December 65% churned against a predicted 39%. Recommendation: send the campaign (the threshold is far below either number), check labels on 1 March 2012, compare with last December before retraining.
- The chatbot (llama3.2) answered the returns and gift-wrap questions correctly with citations, said "I don't know" to Bitcoin, and **missed Norway**, which the documents cover only as "rest of the world".
- The app runs with and without an LLM.
