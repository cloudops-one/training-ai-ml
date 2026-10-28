# 00.3 · The three kinds of learning: supervised, unsupervised, reinforcement

⏱ about 30 minutes · Prerequisites: 00.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Marketing sends you two requests on the same day:

- *"Tell us which customers will stop buying, so we can send them a voucher."*
- *"Tell us what kinds of customers we have, so we can plan campaigns."*

They sound similar. They are not. For the first, ShopWave's history already contains the answer for every past customer: either they came back or they did not. For the second, there is no answer column anywhere. Nobody has ever labelled customers "bargain hunter" or "gift buyer". The data has to speak for itself.

Those two situations need different kinds of machine learning, and confusing them is the most common mistake in a first ML project. This lesson gives you the one question that separates them.

## 2. The idea

Think about three ways a person learns:

- **With a teacher**: you are shown flashcards with the answer on the back. Picture of a cat, "cat". After enough cards, you can name new animals. This is **supervised** learning: every example comes with the correct answer.
- **Without a teacher**: you are handed a box of 500 photos with no labels and asked to sort them into piles that "go together". You will probably group by colour, or by what is in the photo. Nobody told you the piles; you found structure. This is **unsupervised** learning.
- **By trial and reward**: you learn to ride a bike. Nobody can show you a flashcard for balance. You try, you wobble, you fall (penalty), you stay up (reward), you adjust. This is **reinforcement** learning.

The question that tells them apart:

> **Do we already have the answers for past examples?**
> Yes → supervised. No → unsupervised. No, but we can score each attempt → reinforcement.

## 3. Definition

- **Supervised learning**: learning from examples where each input $X$ comes with its correct output $y$ (the **label**). The model learns $f$ so that $f(X) \approx y$ on new data. Two sub-types:
  - **Regression**: $y$ is a number (spend next month, delivery days). Module 02.
  - **Classification**: $y$ is a category (churn yes/no, fraud/fine, mug/candle/cushion). Module 03.
- **Unsupervised learning**: learning from inputs $X$ only. There is no $y$. The model finds structure: groups (**clustering**), compressed representations (**dimensionality reduction**), unusual points (**anomaly detection**), things that occur together (**association rules**). Module 06.
- **Reinforcement learning (RL)**: an **agent** takes actions in an **environment**, receives **rewards**, and learns a **policy** that maximises total reward over time. Used for game-playing, robotics, and, increasingly, for tuning the behaviour of language models. This course covers the idea, not the implementation; it is rarely the first tool a business needs.

You will also meet **semi-supervised** (a few labels, many unlabeled rows) and **self-supervised** (the data generates its own labels, e.g. "predict the next word", which is how LLMs are trained). Both are variations on the first two.

## 4. Formula

Supervised learning, in one line:

$$\text{find } f \text{ such that } f(X_i) \approx y_i \text{ for all past examples } i$$

Unsupervised learning has no $y$, so the goal changes to something like "find groups such that points in the same group are close together". Module 06 makes that precise.

Reinforcement learning: find a policy that maximises $\sum_t \text{reward}_t$. That is all you need for now.

## 5. By hand

Ten ShopWave problems. For each, decide: **supervised (regression or classification)**, **unsupervised**, or **reinforcement**. The test: is there an answer column in the past data?

| # | Problem | Answer column exists? | Kind |
|---|---|---|---|
| 1 | Predict next month's spend per customer | yes, we know what they spent | Supervised, regression |
| 2 | Flag orders likely to be fraud | yes, past frauds were discovered | Supervised, classification |
| 3 | Group customers into 4 marketing segments | no, nobody labelled segments | Unsupervised, clustering |
| 4 | Which products are bought together? | no label; find co-occurrence | Unsupervised, association |
| 5 | Predict delivery time in days | yes, we know past delivery times | Supervised, regression |
| 6 | Spot unusual transactions nobody has seen before | no, that is the point | Unsupervised, anomaly detection |
| 7 | Decide, order by order, which discount to offer to maximise profit over a year | no fixed answer; each choice gets a measurable result | Reinforcement |
| 8 | Classify product photos as mug / candle / cushion | yes, once someone labels a few thousand | Supervised, classification |
| 9 | Summarise 10,000 reviews into the main complaints | no labels; find themes | Unsupervised (topic discovery), or GenAI |
| 10 | Will this customer open the email? | yes, we log opens | Supervised, classification |

Two subtleties worth noticing:

- Row 8: supervised learning needs labels, and labels often have to be **created** by people first. Labelling is frequently the most expensive part of a project.
- Row 3 vs row 10: the same customer table can feed both an unsupervised and a supervised problem. The *question* decides the kind, not the data.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). On the 20-customer ShopWave table you will:

1. Run a **supervised** model: predict `churned` from the other columns, and score it against the known answers.
2. Run an **unsupervised** model: K-Means with 3 clusters on the same columns, with no target, and look at what the groups are.
3. Watch a 20-line **reinforcement** toy: an agent learns which of three discount levels earns the most, purely from rewards.

Pay attention to the *shape* of each call. Supervised: `fit(X, y)`. Unsupervised: `fit(X)` with no `y`. Reinforcement: a loop of *act → reward → update*.

## 7. On real data

The notebook finishes by listing the columns of the real ShopWave data and asking which supervised targets can be *built* from them (churn, next-quarter spend, return-or-not) and which questions are unsupervised (segments, baskets, anomalies). Lesson 00.5 then does both.

## 8. Where it breaks

- **Labels can lie.** "Churned" depends on how you define it (90 days? 180?). "Fraud" only includes the fraud you caught. A supervised model learns your labelling choices, including the mistakes.
- **Unsupervised results need a human.** K-Means will always give you 3 clusters if you ask for 3, whether or not they mean anything. Interpreting them is your job.
- **Reinforcement learning is expensive to get wrong.** The agent learns by trying things. In a live shop, "trying things" means real discounts on real orders. It is used where simulation is possible or mistakes are cheap.
- **The kind does not fix the model.** Supervised ≠ one algorithm. Within supervised learning you still choose between dozens of models; that is Modules 02–05.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Change the number of clusters in the notebook from 3 to 2 and to 5. Do the groups still make sense? What does this tell you about unsupervised learning?
2. In the supervised cell, delete the `churned` column from `X` if it is not already removed, and explain why leaving it in would be cheating.
3. Write three ShopWave questions of your own and sort them. For each supervised one, say exactly which column is the label and how it would be built.

**Quiz**

1. What one question tells supervised from unsupervised learning?
2. "Predict delivery days" and "predict late or on-time": which is regression, which is classification?
3. Why is labelling often the expensive part of a supervised project?
4. K-Means returned three customer groups. Does that prove ShopWave has three kinds of customers?
5. Give one reason reinforcement learning is rarely the first ML a business deploys.

## 10. Recap

- **Supervised**: past examples come with answers. Regression predicts numbers; classification predicts categories.
- **Unsupervised**: no answers; find groups, compressions, oddities, co-occurrences.
- **Reinforcement**: learn by acting and receiving rewards.
- The question *"do we already have the answers?"* decides the kind; the question you ask, not the dataset, determines it.
- In code: `fit(X, y)` vs `fit(X)` vs a reward loop.

**Next:** [00.4 The 7-step ML workflow](../04-ml-workflow/) shows what an ML project looks like from first question to running model, and which step usually goes wrong.
