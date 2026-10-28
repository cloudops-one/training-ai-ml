# 00.4 · The 7-step ML workflow

⏱ about 30 minutes · Prerequisites: 00.3 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The CEO approves a churn project. Your instinct is to open a notebook, load the orders and start training a model. Six weeks later you have a model with 94% accuracy that nobody uses, because:

- marketing cannot act on it (it predicts churn the day *after* the customer has gone),
- the 94% came from a bug that let the model see the future,
- and the data it needs is not available at the moment the prediction has to be made.

None of these are modelling problems. They are *workflow* problems, and they are where most ML projects actually fail. This lesson gives you the seven steps every project goes through, in order, so you know what to do before you train anything and what to do after.

## 2. The idea

Building an ML system is like building a bridge: the concrete pour (training the model) is the famous part, but it is one step out of many, and the survey and foundations decide whether it stands.

```
   1. FRAME          What decision will this support? What is the target?
        │            What would "good enough" look like? Baseline?
        ▼
   2. GET DATA       Where is it? Is it available at prediction time?
        │            Enough rows? Labels? Permissions?
        ▼
   3. EXPLORE        Look. Plot. Count. Find missing values, outliers,
        │            leaks, surprises. (Module 01)
        ▼
   4. PREPARE        Clean, encode, scale, engineer features.
        │            Split into train / validation / test. (Module 04)
        ▼
   5. MODEL          Start simple. Train. Compare models. Tune.
        │            (Modules 02, 03, 05, 06, 07, 08, 09)
        ▼
   6. EVALUATE       Score on data the model has never seen. Compare
        │            with the baseline. Check errors. Explain. (03, 04, 05)
        ▼
   7. DEPLOY & WATCH Ship it. Monitor. Retrain when the world changes.
                     (Module 10)
        │
        └──────────── back to 1 when the question changes ──────────┐
```

The arrows go down, but in real projects you loop: exploring reveals the question was wrong; evaluating sends you back to preparing. Expect to go around two or three times.

## 3. Definition

The seven steps, with the word you will hear for each:

1. **Frame** the problem. Define the **decision** the model supports, the **target** (exactly which column, computed how), the **unit** (per customer? per order? per day?), the **metric** that measures success, and a **baseline** (what the simplest rule scores).
2. **Get the data.** Identify sources, join them, check you are allowed to use them, and check that every feature will exist *at the moment of prediction*.
3. **Explore** (EDA, exploratory data analysis). Distributions, missing values, outliers, correlations, class balance. Mostly plots and counts.
4. **Prepare.** Clean, impute, encode categories, scale, create features, and **split** the data so that the test set is hidden until the end.
5. **Model.** Train a simple model first, then stronger ones. Tune hyperparameters on validation data, never on the test set.
6. **Evaluate.** Score on the test set once. Compare to the baseline. Look at the worst errors. Check fairness and explainability where it matters.
7. **Deploy and monitor.** Make the model available (API, app, batch job), log its predictions, watch for drift, and plan retraining.

## 4. Formula

No equation, but one rule you will see in every module:

> **Time flows one way.** A feature is only allowed if its value was known *before* the moment the prediction is needed.

Churn example. If you predict on 1 June whether a customer will churn by 1 September, you may use orders up to 31 May. You may not use "number of orders in June", and you certainly may not use "days since last order" computed on 1 September. Break this rule and you get the 94% model that is useless: it is reading the answer off the back of the card.

## 5. By hand

Frame the churn project properly, on paper. Fill in each line before reading the suggested answers.

| Step | Question | A reasonable answer for ShopWave |
|---|---|---|
| Decision | What will someone *do* with the prediction? | Marketing sends a £5 voucher to the 500 customers most likely to churn, once a month |
| Unit | One row = ? | One customer, as of the 1st of each month |
| Target | Exactly what is $y$? | 1 if the customer places **no order in the next 90 days**, else 0 |
| Features | Known on the 1st of the month? | Orders in last 30 / 90 / 365 days, days since last order, average basket, country, months as customer |
| Metric | How do we score? | Of the 500 we contact, how many were really going to churn (precision); of all churners, how many we found (recall) |
| Baseline | Dumbest rule? | "Flag anyone with no order in the last 60 days." Any model must beat this |
| Good enough | When do we stop? | Beat the baseline's precision by 10 points at the same recall |
| Risk | What could go wrong? | Using future data by mistake; vouchers going to people who would have returned anyway |

Thirty minutes of this table saves six weeks. Notice that *no model is mentioned*. The model is step 5.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). It walks the seven steps on the 20-customer table in miniature, one short cell per step, with the step number in the heading. Two cells demonstrate the time rule:

- a "leaky" model that is allowed to see `days_since_last` measured *after* the churn window (it scores suspiciously well),
- the honest version, which scores lower and is the only one worth shipping.

You will see this trap properly in Module 04. Here you just need to recognise its shape.

## 7. On real data

Lesson 00.5 runs all seven steps on the real ShopWave data in one notebook. Read the framing cell at the top of that notebook and check it against the table above.

## 8. Where it breaks

- **Skipping step 1.** The most common failure. A model that predicts the wrong thing, for the wrong unit, at the wrong time, is worthless however accurate.
- **Leakage in step 4.** Any preprocessing that sees the test data (scaling with the full dataset's mean, filling missing values with a global median) quietly breaks evaluation. Module 04 fixes this with pipelines.
- **Evaluating on the training data.** Scores 99%, ships, fails. Always hold data back.
- **Stopping at step 6.** A model in a notebook helps nobody. Step 7 is where value appears, and it is also where models silently rot as customer behaviour changes.
- **One pass only.** Expect to loop. Exploration changes framing; evaluation changes features.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Frame a *different* ShopWave project with the same table: "predict whether an order will be returned". Fill all eight rows.
2. In the notebook, find the leaky feature and explain in one sentence why it leaks. Then propose a version of the same feature that would be legal.
3. Write a baseline rule for "predict next-month spend" and compute its error on the 20-customer table (predict the mean for everyone).

**Quiz**

1. List the seven steps in order.
2. What is the "time flows one way" rule?
3. Why must a baseline be defined before modelling?
4. Which step do most failed ML projects skip?
5. Why is a model that is never deployed worth nothing, even at 99% accuracy?

## 10. Recap

- Seven steps: **frame → get data → explore → prepare → model → evaluate → deploy and monitor**, looping as needed.
- Framing fixes the decision, unit, target, metric and baseline, on paper, before any code.
- Features must be known at prediction time. Leaking the future gives impressive, useless models.
- Evaluate on data the model has never seen; compare with the baseline.
- Value appears at deployment, and models decay, so monitoring is part of the job.

**Next:** [00.5 The whole course in one notebook](../05-whole-course-in-one-notebook/) runs all seven steps, and most of the course's techniques, on the real ShopWave data in 30 minutes, so you know where you are going.
