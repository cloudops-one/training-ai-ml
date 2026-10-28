# 07.P · Mini-project: the December stock forecast

⏱ about 2 hours · Prerequisites: 07.1–07.5 · Notebooks: [project.ipynb](project.ipynb) (your work) · [solution.ipynb](solution.ipynb) (one complete answer)

---

## The brief

It is **Sunday 30 October 2011**. The buying team must confirm December orders with overseas suppliers this week. Last year they ran out of best-sellers in the second week of December and over-ordered everything else.

They need, by tomorrow morning:

1. A **weekly revenue forecast** for the five weeks from 6 November to 4 December 2011.
2. A **range** for each week, so they know how much safety stock to hold.
3. A short, plain-English **recommendation**, including how confident you are.

You have every model from this module and two years of weekly revenue up to 30 October. The last five weeks of the data (6 November – 4 December) are held back. You only look at them at the end, to see how your forecast would have done.

## Tasks

1. **Split.** Train on weeks up to 30 October; hide the 5 weeks after.
2. **Honest selection origins.** Work out which backtest origins can be used for choosing a model without their forecast windows touching the hidden weeks.
3. **Backtest the candidates.** At least seasonal naive, Holt-Winters, SARIMA, GBM direct and GBM recursive, plus one idea of your own. Report mean MAE and something about the bad weeks (worst week, or the 80th-percentile error).
4. **Choose** one model using only the backtest. Fix the rule before you look at the test weeks.
5. **Forecast with a range** for each week. Base the range on your model's own backtest errors (or on prediction intervals you have checked).
6. **The reveal.** Compare with the hidden weeks: MAE, 5-week total error, how many weeks fell inside your range, and how the other candidates would have done.
7. **Recommendation.** A paragraph to the buyers: what to plan for, which items need safety stock, how confident you are and why.

## Rubric (20 points)

| criterion | points |
|---|---|
| Correct time-ordered split; test weeks untouched until Task 6 | 3 |
| Selection origins chosen so no forecast window overlaps the test period, with the reason stated | 3 |
| At least five candidates backtested on identical origins and horizon, including seasonal naive | 3 |
| Model chosen by a rule fixed in advance, and the margin over the runner-up discussed honestly | 3 |
| Weekly forecast with a justified range (method explained, not just printed) | 3 |
| Reveal reported fully: MAE, total error, range coverage, comparison with other candidates | 3 |
| Recommendation in plain English, with numbers, safety-stock advice and stated confidence | 2 |

## Hints

- The lesson backtest's last origin, 16 October, forecasts to 20 November. That overlaps the hidden weeks. Drop it (and any other origin whose window ends after 30 October).
- Module 05.1: averaging two models that make *different* mistakes often beats both. Seasonal naive and recursive GBM are a natural pair.
- SARIMA's 95% intervals covered only 85% of weeks in 07.4. A range built from your model's backtest errors is simpler and easier to explain.
- Judge the forecast the way the buyers will use it: week-by-week accuracy matters, but so does the **5-week total**, because that is what gets ordered.

## What the solution found

The [solution](solution.ipynb) is one way to do it. Its results:

- **Selection.** The average of seasonal naive and recursive GBM wins the selection backtest (MAE £29.8k a week over seven origins), ahead of recursive GBM alone (£32.0k) and seasonal naive (£34.0k).
- **Range.** Each week's range is the forecast ± £46k, the 80th percentile of the model's backtest errors.
- **The reveal.** The 5-week total lands within **1.1%** of actual (£1.284m vs £1.298m), and **all 5 weeks** are inside the range. Week by week, though, its MAE of £13.6k ranks only fourth of six. Plain seasonal naive is best (£9.9k) and recursive GBM, the backtest runner-up, is worst (£23.5k).

The lesson is not "model X is best". The backtest ranked candidates within a few thousand pounds of each other, which is less than one typical week's error, so any particular five weeks can reorder them. Every candidate's 5-week total was within 3.5% of actual. With two years of history, last year's pattern plus honest ranges is the right plan, and the models' main job is to confirm it and size the safety stock.

## Stretch tasks

1. **Product level.** Take the 200 products with the highest December 2010 revenue. Build one *global* gradient-boosting model on a stacked product × week table (product's own lags, product ID or category, week of year), and compare its forecast for the top 20 products with per-product seasonal naive.
2. **Bank holidays.** Add a "number of UK bank holidays in this week" feature to the recursive GBM and re-run the selection backtest. Does the 1 May 2011 week stop being the worst?
3. **Calibrated ranges.** Instead of one fixed band, use a separate 80th-percentile error for each horizon (1 to 5 weeks ahead). Are later weeks given wider ranges, as they should be?
