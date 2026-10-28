# 11.3 · Writing it up: README, portfolio, presentation

⏱ about 2 hours · Prerequisites: [11.2](../02-capstone/) well under way

---

## 1. The problem

A reviewer (a hiring manager, a team lead, the head of marketing at ShopWave) gives your project about two minutes. In that time they open the README, look for what you achieved, and decide whether to read further. Most capstones fail this test not because the work is weak, but because the result is buried under setup instructions and library lists.

## 2. The idea

Write for the busiest person who will read it. Put the answer first, then the evidence, then the details. Journalists call this the inverted pyramid; it works for projects too.

## 3. A README that gets read

In this order:

1. **Title that says what it does.** "Ranking customers for the premium catalogue", not "Capstone Project" or "ML-Final-v3".
2. **One-sentence summary** with the result: *"A model that ranks ShopWave customers by their chance of spending £500+ next quarter; for the same 500 catalogues it reaches 405 high-value customers instead of 373."*
3. **The problem** in two or three sentences, in the user's language.
4. **Result table**: baseline and final model side by side, one metric that matters.
5. **What drives it and where it fails**: two or three bullets each.
6. **Data**: source, licence, size, how to download.
7. **Approach**: the steps, briefly, with links to the notebooks.
8. **How to run**: exact commands.
9. **Limitations and next steps.**

The [capstone template README](../02-capstone/capstone_template/README.md) follows this order; the [worked example](../02-capstone/example/README.md) shows it filled in.

## 4. Before and after

**Before** (what most first drafts look like):

> # Capstone
> This project uses machine learning to analyse retail data. I used pandas, numpy, scikit-learn, lightgbm, matplotlib and seaborn. First I cleaned the data, then I did EDA, then I trained several models including logistic regression, random forest and LightGBM and tuned them with GridSearchCV. The best model got 0.815 AUC. See the notebooks for details.
> ## Installation
> pip install -r requirements.txt

What is wrong: no user, no decision, "0.815 AUC" with no baseline means nothing, the library list belongs in `requirements.txt`, and "see the notebooks" asks the reader to do your job.

**After:**

> # Ranking customers for the premium catalogue
> ShopWave posts a £3 catalogue to 500 customers a quarter. This model ranks customers by their chance of spending £500+ in the next three months. **For the same 500 catalogues it reaches 405 high-value customers instead of 373** under the current "highest recent spend" rule.
>
> | Test quarter (Sep–Nov 2011) | AUC | Precision in top 500 |
> |---|---|---|
> | Current rule | 0.744 | 0.746 |
> | **Logistic regression** | **0.815** | **0.810** |
>
> Driven mainly by number of orders and average order value. Misses customers with little history who order big before Christmas; catching them needs data we do not have (browsing, email engagement).

Same project. The second version is shorter and says more.

## 5. Talking about results honestly

- **Always show the baseline next to your number.** A number alone cannot be judged.
- **Say how you split.** "Trained on June, tested on September" is more convincing than "80/20 split".
- **Negative results are results.** "LightGBM did not beat logistic regression (CV AUC 0.847 vs 0.848), so I kept the simpler model" shows judgement. So does "No model beat the seasonal-naive forecast by more than the noise between backtests" (07.2). What reviewers distrust is a project with no limitations section.
- **Don't oversell.** "Predicts customer behaviour" becomes "ranks customers for one decision". "Production-ready" becomes "a batch scoring script and a monitoring plan".
- **Name the uncertainty.** One test quarter is one data point; say so, and say what would make it stronger.

## 6. The 5-minute presentation

Six slides, about 45 seconds each, plus time for one question.

| # | Slide | What goes on it |
|---|---|---|
| 1 | **The problem** | Who, what decision, why it matters. One picture or number. |
| 2 | **The data** | Source, size, what one row is, the split (draw the timeline if it is time-based). |
| 3 | **The approach** | Baseline, the models compared, how the final one was chosen. No code. |
| 4 | **The result** | The table: baseline vs model, in the user's units. One sentence of business impact. |
| 5 | **What drives it, where it fails** | Top features in plain words; the main failure group and why. |
| 6 | **Limits and next steps** | What you would not trust it for; what you would do with another week. |

Practise it out loud with a timer. If it runs over, cut detail from slide 3, never from slides 4 and 6.

## 7. A portfolio, not a pile of repos

Three to five projects you can explain beat twenty forks. A good set shows range:

- **One tabular supervised project** with honest evaluation (your capstone, or the churn work from Modules 03–05).
- **One unsupervised or forecasting project** (segments from Module 06 or the forecast from Module 07).
- **One deep learning or LLM project** (the image classifier from 08.P or the support chatbot from 09.P).

For each: a clear README as above, pinned on your GitHub profile. On your profile README, one line per project with its result. Rewrite the course mini-projects in your own words and with your own variations before you show them; a reviewer may recognise the course.

## 8. GitHub hygiene

Before you make a repository public:

- **No data.** Download scripts only. Some licences forbid redistribution, and large files make the repo unusable.
- **No keys.** Search for them: `git log -p | grep -i -E "api_key|secret|sk-"`. If a key was ever committed, **revoke it at the provider** first; deleting the file is not enough, because it stays in the history.
- **No personal data.** Customer names, emails or addresses must not appear in notebooks or outputs.
- **No giant notebook outputs.** Clear huge printed tables before committing; keep charts.
- **A `.gitignore`** (the template has one) and a licence file (MIT is common for code).
- **Notebooks run top to bottom.** Restart the kernel and run all before every commit.

## 9. Check yourself

1. Give your README to someone for two minutes, then take it away. Ask them: what does it do, for whom, and how much better is it than before? Fix whatever they could not answer.
2. Time your presentation out loud. Under five minutes?
3. Run the hygiene checks in section 8 on your repository.

## 10. Recap

- Answer first: title, one-sentence result, baseline vs model table.
- Honest beats impressive: show the baseline, say how you split, keep the limitations.
- Six slides, five minutes, practised out loud.
- No data, keys or personal information in public repositories.

**Next:** [11.4 Interview preparation](../04-interview-prep/).
