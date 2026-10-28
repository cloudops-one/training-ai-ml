# 07.2 · Baselines and honest evaluation

⏱ about 1.5 hours · Prerequisites: 07.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A supplier's sales rep has a forecasting tool to sell you. It is "AI-powered" and "93% accurate". The buying team is impressed and wants to sign.

Before anyone signs anything, you need answers to two questions. *Accurate compared with what?* And *measured how?* A forecast is only useful if it beats something cheap and obvious. Without a baseline and a fair test you cannot tell whether a tool is clever or just lucky with the weeks it was scored on.

This lesson builds both. First come four baselines so simple you can compute them on paper. Then comes a rolling backtest, which tests any forecaster the way the future will: one origin at a time, never peeking ahead.

## 2. The idea

Weather forecasters have a humbling benchmark: *"tomorrow will be like today"*. It is right surprisingly often, and any forecasting service that cannot beat it is worthless.

ShopWave's equivalent is *"this week will be like the same week last year"*. You saw in 07.1 that the year-ago week is the strongest clue there is (autocorrelation 0.81). So before building anything clever, we score the obvious guesses. Then we hold every later model to the same test.

Testing has a time rule too. When you forecast in late October, you know nothing about November. A fair test recreates that: pretend it is a past date, forecast forward, check against what really happened, and repeat for many past dates.

## 3. Definition

**The four baselines** (*h* = how many steps ahead, *m* = season length):

- **Naive**: every future value equals the last observed value.
- **Mean**: every future value equals the average of all history.
- **Seasonal naive**: each future value equals the value one season earlier (same week last year).
- **Drift**: a straight line from the first to the last observation, extended forward.

**Error metrics:**

- **MAE**: mean absolute error, in pounds. The headline metric in this module.
- **RMSE**: root mean squared error; punishes big misses more.
- **MAPE**: mean absolute percentage error; scale-free, but breaks when actual values are near zero.
- **sMAPE**: a "symmetric" MAPE that divides by the average of actual and forecast; less explosive, still awkward.
- **MASE**: MAE divided by the in-sample MAE of seasonal naive; 1.0 means "as good as copying last season".

**Evaluation:**

- **Forecast origin**: the date you stand on when making the forecast. Only data up to the origin may be used.
- **Horizon**: how far ahead you forecast (here, 5 weeks).
- **Rolling-origin backtest**: repeat "train up to origin, forecast *h*, score" at several origins and average the errors.

## 4. Formula

With history $y_1, \dots, y_T$, season length $m$ and horizon $h$:

$$\text{naive: } \hat y_{T+h} = y_T \qquad \text{mean: } \hat y_{T+h} = \bar y \qquad \text{seasonal naive: } \hat y_{T+h} = y_{T+h-m}$$

$$\text{drift: } \hat y_{T+h} = y_T + h \cdot \frac{y_T - y_1}{T - 1}$$

Errors over $n$ forecast points:

$$\text{MAE} = \frac{1}{n}\sum |y - \hat y| \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum (y - \hat y)^2} \qquad \text{MAPE} = \frac{100}{n}\sum \frac{|y - \hat y|}{|y|}$$

$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{T-m}\sum_{t=m+1}^{T} |y_t - y_{t-m}|}$$

## 5. By hand

Eight weeks of history (£k) with a 4-week season, and the 4 weeks that followed:

| week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | **10** | **11** | **12** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| revenue | 10 | 12 | 18 | 14 | 11 | 13 | 20 | 15 | **12** | **15** | **22** | **17** |

**Forecasts for weeks 9–12:**

- Naive: 15, 15, 15, 15.
- Mean: (10+12+18+14+11+13+20+15) / 8 = 113 / 8 = **14.125** each week.
- Seasonal naive: copy weeks 5–8 → **11, 13, 20, 15**.
- Drift: slope = (15 − 10) / 7 = 0.714 → 15.71, 16.43, 17.14, 17.86.

**Seasonal naive, scored:**

| week | actual | forecast | error | error² | % error |
|---|---|---|---|---|---|
| 9 | 12 | 11 | 1 | 1 | 1/12 = 8.33% |
| 10 | 15 | 13 | 2 | 4 | 2/15 = 13.33% |
| 11 | 22 | 20 | 2 | 4 | 2/22 = 9.09% |
| 12 | 17 | 15 | 2 | 4 | 2/17 = 11.76% |
| | | **mean** | **1.75** | **3.25** | **10.63%** |

MAE = 1.75, RMSE = √3.25 = **1.80**, MAPE = **10.63%**.

The same arithmetic for the others gives MAE 3.00 (naive), 3.44 (mean), 2.71 (drift). Seasonal naive wins, because the series repeats.

**MASE.** In the training weeks, seasonal naive's errors were |11−10|, |13−12|, |20−18|, |15−14| = 1, 1, 2, 1 → mean **1.25**. So MASE for seasonal naive = 1.75 / 1.25 = **1.40**: a bit worse in the test than it typically was in training.

**How MAPE breaks.** Three weeks: actual £250k, £260k, and a Christmas week of £1k. Forecasts miss by £10k, £10k and £29k. MAE is a sensible **£16,333**. MAPE is **969%**, because the Christmas week alone is a 2,900% error. If any week is exactly zero, MAPE is infinite (or undefined).

## 6. In code

```python
def seasonal_naive(train, h):
    return train.iloc[-52:-52 + h].values         # same weeks, one year earlier

def backtest(forecast_fn, series, origins, h=5):
    rows = []
    for origin in origins:
        train = series[series.index <= origin]    # only the past
        test = series[series.index > origin].iloc[:h]
        pred = forecast_fn(train, len(test))
        rows.append(np.abs(test.values - pred).mean())
    return np.mean(rows)
```

The notebook defines all four baselines and `backtest()` once; lessons 07.3–07.5 and the project reuse them unchanged so every model is judged on identical terms.

## 7. On real data

**One split** (origin 1 May 2011, 5 weeks ahead): seasonal naive MAE **£22.9k**, mean £28.8k, naive £77.5k, drift £84.0k. The last week of April 2011 had two bank holidays and sold only £73k, so naive and drift copied that bad week forward.

**The rolling backtest** (8 origins every 4 weeks from 3 April to 16 October 2011, horizon 5 weeks, 40 forecast-weeks in total):

| baseline | mean MAE | worst origin | times best |
|---|---|---|---|
| **seasonal naive** | **£32,450** | £75,249 | 4 of 8 |
| naive | £42,200 | £77,487 | 1 |
| drift | £43,037 | £84,013 | 1 |
| mean | £49,412 | £104,027 | 2 |

Seasonal naive has the lowest average error, but it is not best everywhere. Its worst origin is 18 September, when the 2011 autumn ramp-up started at a different pace from 2010. Even the plain mean wins two origins in quiet summer weeks. That is why you average over many origins: any single one can mislead.

**£32,450 per week is the bar.** Weekly revenue averages about £164k, so seasonal naive is typically off by about a fifth. Every later model must beat this in the same backtest to earn its place.

## 8. Where it breaks

- **Calendars move.** Seasonal naive copies "week 17 last year", but Easter, bank holidays and Christmas weekdays shift. The week ending 1 May 2011 (two bank holidays) is off by 133% against its 2010 twin.
- **Level changes.** If the business grew 30% this year, seasonal naive under-forecasts every week. (Exercise 4 tries a level adjustment; on ShopWave it does not help, because 2011 runs at about the same level as 2010.)
- **MAPE on low or zero weeks.** Christmas weeks make it explode. Use MAE (or MASE when comparing series of different sizes).
- **Few origins, few seasons.** Eight origins over seven months is better than one split, but none of them tests a November with a model trained on a previous November. With two years of data that test is simply not possible.
- **Backtest ≠ future.** It estimates typical error under past conditions. A new competitor, a pandemic or a website change can break every method at once.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: naive, seasonal-naive and drift forecasts and MAE for `[20, 24, 30, 26, 22, 25, 33, 27]` → `[23, 27, 34, 28]` (season 4).
2. Score seasonal naive over every week of 2011 with MAE and MAPE. What happens to MAPE, and which week has the largest percentage error?
3. Re-run the backtest with a 10-week horizon. Do the rankings change?
4. Build a level-adjusted seasonal naive and backtest it. Does it win?

**Quiz**

1. What does seasonal naive forecast for next week, for ShopWave?
2. Why does this module report MAE rather than MAPE?
3. In the by-hand table, what would seasonal naive forecast for week 11, and why?
4. Why is one train/test split not enough to rank forecasters?
5. In a rolling backtest, may a model's training data include anything after its origin?

## 10. Recap

- Always start with baselines: naive, mean, seasonal naive and drift cost nothing and set the bar.
- MAE in pounds is the clearest metric for buyers; MAPE breaks on quiet weeks, and MASE compares against seasonal naive.
- A rolling-origin backtest repeats "train on the past, forecast, score" at many dates, which is the honest way to test a forecaster.
- On ShopWave's weekly revenue, seasonal naive wins the backtest with MAE about £32k per week.
- Every model from here on must beat that number in the same backtest.

**Next:** [07.3 Smoothing and Holt-Winters](../03-smoothing-holt-winters/) builds the first real forecasting model and puts it through this test.
