# 07.1 · Time series: trend, seasonality, noise

⏱ about 1.5 hours · Prerequisites: Module 01 (Pandas, dates, plotting), 04.1 (train/test splits) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

It is late October. The buying team must place December orders with suppliers this week, because Christmas stock ships from overseas and takes five weeks to arrive. Last year they guessed. They ran out of the best-sellers in the second week of December and were left holding everything else in January.

They ask you: *"How much will we sell each week from now until Christmas?"*

Every model you have built so far predicts a target from *other columns*: churn from recency, spend from visits. Here there are no other columns. There is one number per week, and the only thing you know about next week is what happened in the weeks before it. That is a **time series**, and it needs its own tools, starting with simply *looking* at it properly.

## 2. The idea

Think about the temperature in your city. If you plotted it every day for two years you would see three things layered on top of each other:

- a **slow drift** (this decade is a bit warmer than the last),
- a **yearly wave** (summer up, winter down, every year, same shape),
- **day-to-day jitter** (a cold Tuesday in July).

Sales are the same. Pull the three layers apart and each becomes easy to reason about: the drift tells you whether the business is growing, the wave tells you how much more to stock in November, and the jitter tells you how wrong you should expect to be.

```
observed   =   trend        +   seasonality        +   noise
  /\/\_/\      ______/‾‾‾       /\    /\    /\          . ' . , '
```

## 3. Definition

- **Time series**: a sequence of values recorded at regular time steps (daily, weekly, monthly). Order matters; shuffling destroys it.
- **Frequency**: the step size. Changing it is **resampling**: daily → weekly sums seven days into one value.
- **Rolling mean (moving average)**: the average of the last *k* values, recomputed at every step. It smooths out noise.
- **Trend**: the long-run level of the series.
- **Seasonality**: a pattern that repeats on a fixed calendar, with a known **period** (7 days for a week, 52 weeks or 12 months for a year).
- **Noise / residual**: what is left when trend and seasonality are removed.
- **Additive** decomposition: observed = trend + seasonal + residual (the seasonal swing is a fixed number of pounds).
- **Multiplicative** decomposition: observed = trend × seasonal × residual (the swing is a percentage).
- **Autocorrelation at lag *k***: the correlation between the series and itself shifted *k* steps back. "Does this week look like *k* weeks ago?"

## 4. Formula

Rolling mean with window *k* at time *t*:

$$\text{MA}_t = \frac{1}{k}\sum_{i=0}^{k-1} y_{t-i}$$

Additive and multiplicative decomposition:

$$y_t = T_t + S_t + R_t \qquad\qquad y_t = T_t \times S_t \times R_t$$

- $y_t$: the value at time $t$ (one week's revenue).
- $T_t$: trend, estimated by a *centred* rolling mean over one full period.
- $S_t$: seasonal effect, the average of $y_t - T_t$ (or $y_t / T_t$) for that position in the year.
- $R_t$: residual, whatever remains.

Autocorrelation at lag $k$ is the ordinary correlation from lesson 01.8, computed between $y_t$ and $y_{t-k}$.

## 5. By hand

One week of ShopWave revenue, in £ thousands (made-up round numbers with the real pattern: Thursday busiest, Saturday closed):

| day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| revenue | 30 | 32 | 28 | 40 | 25 | 0 | 18 |

**3-day rolling mean.** Average each day with the two before it.

| day | window | 3-day mean |
|---|---|---|
| Mon | — | — |
| Tue | — | — |
| Wed | 30, 32, 28 | 90 / 3 = **30.00** |
| Thu | 32, 28, 40 | 100 / 3 = **33.33** |
| Fri | 28, 40, 25 | 93 / 3 = **31.00** |
| Sat | 40, 25, 0 | 65 / 3 = **21.67** |
| Sun | 25, 0, 18 | 43 / 3 = **14.33** |

Two things to notice. The first two days have no value, because three values do not exist yet. And Saturday's closure drags the mean down for three days running. A rolling mean over **seven** days would swallow the weekly rhythm completely: (30 + 32 + 28 + 40 + 25 + 0 + 18) / 7 = 173 / 7 = **24.71**. That is exactly what weekly resampling does, which is why this module forecasts weekly totals.

**Decomposition in one line.** If the trend for a November week is £158k and the seasonal effect for November is +£111k, the model "expects" £269k; if the week actually sold £276k, the residual is +£7k. That is all decomposition is: splitting each number into *usual level*, *usual for this time of year*, and *surprise*.

## 6. In code

```python
daily   = rev.resample("D").sum()      # every day, 0 when closed
weekly  = rev.resample("W").sum()      # weeks ending Sunday
monthly = rev.resample("ME").sum()     # labelled by month-end

daily.rolling(28).mean()               # 28-day moving average

from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(weekly, model="additive", period=52)
dec.trend, dec.seasonal, dec.resid

weekly.autocorr(52)                    # this week vs the same week last year
```

## 7. On real data

The notebook joins both years of Online Retail II (1 December 2009 to 4 December 2011). That gives 734 days, 105 weeks and 24 months.

Three cleaning decisions are made before anything else, and each one matters for forecasting:

- **Overlap removed.** The two Excel sheets share the first nine days of December 2010.
- **Two one-off orders dropped.** One was 80,995 paper birdies and the other 74,215 storage jars, and both were cancelled the same day. Otherwise they would appear as £168k and £77k phantom spikes.
- **Series cut at Sunday 4 December 2011.** The raw data stops on a Friday. A partial final week, or a four-day "December", would look like a collapse in sales.

What the data shows:

- **Weekday rhythm.** Thursday is the busiest day (£36k on an average open day), Sunday the quietest (£18k), and Saturday is closed.
- **Seasonality, weekly.** The additive decomposition puts January to August £12k–£59k a week *below* trend and October–November about £100k a week *above* it. The trend itself barely moves (£154k–£162k a week): ShopWave is not growing, it is very seasonal.
- **Seasonality, monthly.** The multiplicative factors are November **1.76×** an average month, October 1.55×, and February 0.66×. That one table is most of a stock plan. Multiplicative decomposition fails on the weekly data, because the Christmas-closure weeks are zero.
- **Autocorrelation.** It is 0.67 at lag 1 (last week) and **0.81 at lag 52** (the same week last year). The year-ago week is the single best clue you have.
- **Random splits are misleading.** Predicting each randomly hidden week from its two neighbours gives MAE £41,766. Predicting the last 20% of weeks from the last known week gives £78,609. The random split looks twice as good only because it peeks at the following week.

## 8. Where it breaks

- **Partial periods at the edges.** The first and last day, week or month are often incomplete. Always check before plotting, and never feed a partial period to a model.
- **One-off events look like signal.** A single giant order, a website outage or a pandemic will be "learned" as part of the pattern unless you spot and handle it.
- **Two years is very little for a yearly season.** The centred 52-week trend is missing for the first and last 26 weeks, and every seasonal estimate rests on just two Novembers. Treat the numbers as rough.
- **Calendar effects move.** Easter changes date; Christmas falls on a different weekday each year. Week 47 in 2010 and week 47 in 2011 are not quite the same week.
- **Additive vs multiplicative is a judgement.** If the business doubled in size, a fixed £100k November bump would be wrong; a 1.76× factor would still be right. Check whether the swing grows with the level.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand, then in code: the 3-day rolling mean of `[12, 15, 9, 0, 21, 18]`.
2. Find the best and worst month of each year. How many times bigger is the best? Which months are partial?
3. What share of revenue does each weekday take? Which two days would you add warehouse staff on?
4. Plot the ACF of the *daily* series for 30 lags. Which lags stand out, and what does that suggest for a daily model?

**Quiz**

1. What does `resample("W").sum()` do, and what must the index be?
2. Why is the 3-day rolling mean empty on the first two days?
3. Name the three components of a time series and give a ShopWave example of each.
4. Why can't you use a multiplicative decomposition on ShopWave's weekly series?
5. Why is a random train/test split misleading for forecasting?

## 10. Recap

- A time series is one value per time step; order is information, so never shuffle it.
- `resample` changes the zoom level; weekly totals remove the closed-Saturday rhythm and are ShopWave's natural forecasting unit.
- Rolling means smooth noise; decomposition splits a series into trend + seasonality + residual (or ×).
- ShopWave has a flat trend and a huge yearly season: November is about 1.76× an average month.
- Autocorrelation at lag 52 (0.81) says "same week last year" is the strongest clue, and only time-ordered splits test a forecast honestly.

**Next:** [07.2 Baselines and honest evaluation](../02-baselines-and-evaluation/) turns "same week last year" into a forecast and builds the rolling backtest every later model must beat.
