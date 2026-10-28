# 00.5 · The whole course in one notebook

⏱ about 1.5 hours · Prerequisites: 00.1–00.4 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You have a map (00.2), you know the three kinds of learning (00.3) and the seven steps (00.4). What you do not have is a feel for what a finished project *looks like*. Reading about bridges is not the same as walking across one.

So, before learning any technique properly, you will run a complete ML project on ShopWave's real data: 540,000 transactions, a churn model, customer segments, a forecast and a saved model. In about thirty minutes. Without understanding most of the code.

That last part is deliberate.

## 2. The idea

When you learn a language, hearing a whole conversation first, even without understanding it, makes every later grammar lesson land: you know what it is *for*. This notebook is the conversation. Every module after this one will answer a question you already have, because you have seen the step happen and did not understand it.

Rule for today: **run, read the text, look at the output, move on.** If a line of code puzzles you, note the module that teaches it and keep going.

## 3. Definition

What the notebook builds, in the vocabulary from 00.3 and 00.4:

- **Unit**: one customer, as of a **cut-off date** (2011-09-09, 90 days before the data ends).
- **Features** (from before the cut-off): recency (days since last order), frequency (orders), monetary (total spent), average basket, number of distinct products, tenure, UK or not. Together the first three are called **RFM**, the oldest customer-scoring method in retail.
- **Target** (from after the cut-off): `churned` = 1 if no order in the following 90 days.
- **Supervised model**: a **random forest**, hundreds of decision trees voting.
- **Unsupervised model**: **K-Means** on RFM, three segments.
- **Forecast**: a **seasonal naive** estimate for December 2011.
- **Deployment**: the forest saved with `joblib` and reloaded to score a new customer.

## 4. Formula

Only one, and you already know it from 00.4: **time flows one way.**

$$\text{features} = f(\text{orders before cut-off}) \qquad \text{label} = g(\text{orders after cut-off})$$

Nothing from the right-hand side is allowed to touch the left. The cut-off date is the fence.

## 5. By hand

Before running anything, predict what you will see. Write down your guesses:

1. What fraction of customers will churn under this definition? (Think: a gift retailer, three autumn months before Christmas.)
2. Will the random forest beat the baseline "flag anyone with no order in the last 60 days"? By how much?
3. Which feature will matter most: recency, frequency or money?
4. What will the three segments look like?

Then compare with the real numbers. The actual results:

| | Guess | Actual |
|---|---|---|
| Churn rate | | 42.9% of 3,362 customers |
| Baseline accuracy | | 61.5% |
| Forest accuracy | | 66.2% |
| Top features | | monetary and recency, then number of products; country almost irrelevant |
| Segments | | loyal big spenders (10% churn), occasional (33%), lapsed (61%) |

A modest model. That is what honest first results look like, and a big part of this course is learning to tell *modest and honest* from *impressive and leaky*.

## 6. In code

Open [lesson.ipynb](lesson.ipynb) and run it top to bottom (about one minute of computation after the one-time download). The sections, each labelled with the module that teaches it:

| Section | What happens | Module |
|---|---|---|
| Step 2 | Download and load 540k rows | 01 |
| Step 3 | Clean: drop cancellations, guests, bad prices; chart monthly revenue | 01 |
| Step 4 | Build the customer table with RFM features and the churn label; split train/test | 01, 04 |
| Step 5 | Train a random forest | 03, 05 |
| Step 6 | Accuracy vs baseline; confusion matrix; feature importance | 03, 05 |
| Detour | K-Means segments, profiled | 06 |
| Detour | Naive December forecast | 07 |
| Step 7 | Save, reload, predict for a new customer | 10 |

## 7. On real data

This whole notebook *is* the real data. One thing to notice while it runs: the cleaning step throws away over a quarter of the rows (guest checkouts have no customer ID; cancellations have negative quantities). Real data is like this. Module 01 teaches you to make those decisions deliberately instead of by accident.

## 8. Where it breaks

Everything in this notebook is a first draft, and you will learn why each of these is a problem:

- **Accuracy is the wrong metric** for a 43/57 split where the two mistakes cost different amounts. (03.6, 03.7)
- **One train/test split** on 841 customers gives a noisy score. (04.7, cross-validation)
- **The forest is untuned** and unexplained beyond a bar chart. (05.6, 05.8)
- **K-Means with k=3** was a guess. (06.3)
- **The forecast is a single number with no uncertainty** and no evaluation. (07.2)
- **"Deployment" is a file on disk.** Nobody can call it. (10.2, 10.3)

If you reach the end of the course and can list what is wrong with this notebook without looking at this section, you have learned what you came for.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Move the cut-off to 2011-06-09 and re-run Steps 4–6. What changes and why?
2. Try baselines at 30, 90 and 120 days of recency. How close does the best one get to the forest?
3. Ask K-Means for four segments and name them in business language.

**Quiz**

1. Why was the cut-off date set 90 days before the end of the data?
2. What is the baseline, and why does it need no model?
3. In the confusion matrix, what does the top-right square count?
4. Which features did the forest rely on most, and which did it ignore?
5. Which model in the notebook received no label at all?

## 10. Recap

- A complete project: frame, load, clean, build features behind a cut-off, split, train, evaluate against a baseline, explain, segment, forecast, save, serve.
- First honest results are modest: 66% vs a 62% baseline. Most of the course is about doing each step properly so that number is both higher and trustworthy.
- Every section mapped to a module. When a later lesson feels abstract, come back here and find the cell it improves.

**Next:** [Module 01 · Working with Data](../../01-working-with-data/). The first thing this notebook did was load and clean a table. That is where the course proper begins.
