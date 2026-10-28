# 11.2 · The capstone

⏱ about 15 hours · Prerequisites: [11.1](../01-choosing-a-problem/) with a completed scoping sheet · Starter: [capstone_template/](capstone_template/) · Worked example: [example/](example/)

---

## 1. The brief

Take the problem you scoped in 11.1 from raw data to something a real user could act on, and present it so that someone who has never seen this course understands it in five minutes.

There is no scaffold. The [template](capstone_template/) gives you an empty structure and prompts; the [worked example](example/) shows one finished answer on ShopWave data. Draft your own before you read the example closely.

## 2. Required deliverables

| # | Deliverable | Where in the template | Taught in |
|---|---|---|---|
| 1 | A public repository with a README that states the problem, the result and how to run it | `README.md` | 11.3 |
| 2 | A reproducible environment | `requirements.txt` | [setup](../../../setup/README.md) |
| 3 | A data acquisition script (data itself not committed) and a data README with licence and dictionary | `src/download_data.py`, `data/README.md` | 01.3, 04.2 |
| 4 | An EDA notebook with a written conclusion under every chart | `notebooks/01_eda.ipynb` | 01.9 |
| 5 | A baseline: the dumbest model and, if one exists, the current business rule | `notebooks/02_baseline.ipynb` | 02.1, 03.1 |
| 6 | **At least two modelling approaches**, compared with cross-validation on training data only | `notebooks/03_models.ipynb` | 04.7, 04.8, 05.10 |
| 7 | One honest test-set result against the baseline | `notebooks/03_models.ipynb` | 04.1 |
| 8 | Error analysis: who or what does the model get wrong, and why | `notebooks/03_models.ipynb` | 03.6, 08.8 |
| 9 | Explainability: global and for two individual predictions | `notebooks/03_models.ipynb` | 05.8 |
| 10 | **Either** a working demo (FastAPI endpoint or Streamlit app) **or** a written deployment plan | `src/app.py` or a section in the README | Module 10 |
| 11 | A monitoring plan: what you measure, how often, what triggers a retrain | README or model card | Module 10 |
| 12 | A model card | `model_card.md` | 05.10 |
| 13 | A 5-minute presentation (slides or a recorded walkthrough) | anywhere you like | 11.3 |

For a generative AI capstone (a RAG assistant, an LLM classifier), deliverables 5–9 become: a written evaluation set built **before** the system, at least two configurations compared on it (for example two chunk sizes, two prompts or two models), per-question failure analysis, and retrieval evidence for answers (09.10, 09.11).

## 3. Rubric (100 points)

Score yourself honestly before you ask anyone else to.

| Area | Points | Excellent (full marks) | Good (about two-thirds) | Needs work (about a third or less) |
|---|---|---|---|---|
| **Problem framing** | 10 | User, decision, target, unit, baseline and metric all stated, and the metric fits the decision | Most stated; metric reasonable but not justified | Vague goal; metric chosen by habit (e.g. accuracy on rare classes) |
| **Data and reproducibility** | 10 | One command recreates the data; licence and dictionary documented; repo runs from a fresh clone | Runs with small fixes; documentation partial | Data committed or missing; does not run elsewhere |
| **EDA** | 10 | Charts each end in a conclusion; target, leakage, outliers and drift checked | Good charts, few conclusions | Charts without interpretation |
| **Honest evaluation** | 15 | Split matches how the model will be used (often by time); no leakage; test set used once; baseline beaten or honestly not beaten | Sound split, minor leakage risk not discussed | Tuned on the test set, leakage, or no baseline |
| **Modelling** | 15 | Two or more approaches in pipelines, compared with CV mean ± spread; choice justified by more than the top score | Two approaches compared, choice by score alone | One model, or comparison not like-for-like |
| **Error analysis and explanation** | 15 | Concrete failure groups with a cause and a fix (data, features or "not fixable"); global and local explanations in plain words | Some error analysis, explanations not tied to the decision | Feature importance plot only |
| **Deployment and monitoring** | 10 | Working demo or a specific deployment plan, plus a monitoring plan with triggers | Demo works but no monitoring, or a generic plan | Missing |
| **Communication** | 15 | README makes sense in 2 minutes; model card complete; 5-minute talk covers problem → result → limits | Complete but long or jargon-heavy | Reader cannot tell what was achieved |

Pass: 60. Portfolio-ready: 80. A project that honestly fails to beat its baseline and explains why can still score above 80.

## 4. Milestones by hours

Hours, not dates: spread them however your week allows.

| After about… | You should have | If you are behind |
|---|---|---|
| 1 h | Scoping sheet complete, data downloaded and loading | Rescope (11.1); do not continue with data you cannot load |
| 3 h | EDA notebook with conclusions; modelling table built | Drop secondary EDA; build the table |
| 5 h | Baseline scored; first model end to end, however rough | This is the most important milestone. Everything after it is improvement |
| 9 h | Two approaches compared with CV; final model chosen; test set used once | Stop tuning. Two simple models compared fairly beat one tuned to death |
| 12 h | Error analysis and explanations written | Keep to two failure groups and two local explanations |
| 14 h | Demo or deployment plan; monitoring plan; model card | Choose the written plan over a half-working demo |
| 15 h | README, presentation, a run from a fresh clone | Cut scope, not the write-up |

## 5. Self-review checklist

Before you call it done:

- [ ] A fresh clone, a new virtual environment and the commands in the README reproduce the results.
- [ ] No data, model files or API keys are committed (`git log --stat` to check).
- [ ] Every feature is available at prediction time (the data dictionary says so).
- [ ] Preprocessing is inside the pipeline, so cross-validation is leak-free.
- [ ] The test set was used once, after the model was chosen.
- [ ] The baseline appears in the results table next to the final model.
- [ ] Every chart has a sentence saying what it shows.
- [ ] Numbers in the README match the notebooks.
- [ ] Limitations are written down, including at least one that a critic would raise.
- [ ] The presentation fits in five minutes when you time it out loud.

## 6. Getting feedback

- **Swap with a peer.** Give them only the repository link and the rubric. Ask them to score it and to write down the first question they would ask. That question is the gap in your README.
- **Explain it to a non-technical friend.** If they can tell you who uses it and what changes, your framing works.
- **Post it.** Share the repository in a learning community (Kaggle discussions, a local meetup, a study group) and ask one specific question, for example "Is my time-based split correct for this problem?".
- **Revisit after a week.** Read the README cold. Fix whatever confused you.

## 7. Recap

- Thirteen deliverables, scored out of 100. Honest evaluation and clear communication are worth as much as modelling.
- Hit the 5-hour milestone (a rough end-to-end result) before polishing anything.
- Compare your work with the [worked example](example/) only after your own draft exists.

**Next:** [11.3 Writing it up](../03-writing-it-up/).
