# 04.P · Mini-project: rebuild churn as a leak-free pipeline

⏱ about 1 hour (longer if you write everything yourself) · Prerequisites: all of Module 04 · Notebooks: [project.ipynb](project.ipynb) (your work) · [solution.ipynb](solution.ipynb) (one good answer)

---

## The brief

The retention manager wants to use your churn model for real: on **1 September 2011** they will score every active customer and send win-back vouchers to the ones most likely to leave in the next three months.

Before they spend money on it, the head of data asks one question: *"How do I know your number is real?"* Your Module 03 model was built on a random split of customers, preprocessed in loose notebook cells and tuned by peeking at the test set. This project rebuilds it the way it would be built for production, so that the score you report is the score they will get.

## What "leak-free" means here

The model must be trained exactly as it will be used: **on the past, to predict the future**.

```
Dec 2010 ──────────── 1 Jun 2011 ──── Jun–Aug ──── 1 Sep 2011 ──── Sep–Nov ──── Dec 2011

          features for TRAINING   │ training label │
          ───────────────────────►│◄──────────────►│
                                                    │
          features for TEST (everything up to 1 Sep)│  test label  │
          ─────────────────────────────────────────►│◄────────────►│
```

- **Training table**: customer features from orders before 1 June; label = no order June–August.
- **Test table**: customer features from orders before 1 September; label = no order September–November. This is the same 3,317-customer table used in Modules 01.9–04.
- Nothing from September onwards is touched until the very last step.

## Tasks

`project.ipynb` loads the data and gives you `build_table(cutoff)`. You do the rest:

1. **Time-based split.** Build the training table (cut-off 2011-06-01) and the test table (cut-off 2011-09-01). Report sizes and churn rates.
2. **A quick look.** Churn rate by recency band, on the training table only.
3. **Preprocessing.** A `ColumnTransformer`: numeric columns imputed, logged and scaled; `country` imputed and one-hot encoded with rare countries grouped.
4. **Compare three models** (logistic regression, KNN *k* = 101, decision tree depth 5), each in a `Pipeline`, by 5-fold stratified CV AUC on the training table. Report mean ± std.
5. **Choose and tune.** Justify your choice; tune one hyperparameter by CV.
6. **One final test.** Fit on the full training table; score **once** on the test table: AUC, accuracy, confusion matrix, churn rate among the top 500 predicted.
7. **Leakage audit.** Complete the five-row table in the notebook.
8. **Conclusion.** Five or six sentences for the retention manager.

## Rubric (20 points)

| Area | Points | Full marks when… |
|---|---|---|
| Time-based split | 4 | Two tables at the right cut-offs; sizes and churn rates reported; the difference in churn rate noticed |
| Pipeline | 4 | All preprocessing inside one `Pipeline` / `ColumnTransformer`; nothing fitted outside it; unknown countries handled |
| Model comparison | 4 | Three models compared with stratified CV on training data only; mean ± std shown; choice justified by more than the top score |
| Final test | 3 | Test set used exactly once, after all choices were made; AUC, accuracy vs baseline and top-500 churn rate reported |
| Leakage audit | 3 | Each of the five risks addressed specifically, with reference to the code |
| Conclusion | 2 | Plain language; says how sure you are, how to use the scores and what to do next |

## Stretch goals

- Add one engineered feature from lesson 04.6 and show, with CV fold by fold, whether it really helps.
- Train on the cut-off 2011-03-01 as well, and compare how well each training date transfers to September.
- Use L1 logistic regression (lesson 04.10) and report which features it keeps.

## What a good result looks like

The solution's logistic regression scores **CV AUC 0.752 ± 0.020** on the June training table and **test AUC 0.741** on September: close, so the evaluation is honest. Among the 500 customers it rates most at risk, 66.8% really churned (43.6% overall). Your numbers will differ a little if you make different choices; what matters is that your CV estimate and your final test score agree and that you can explain why nothing leaked.
