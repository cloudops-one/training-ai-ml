# Module 07 · Forecasting

**ShopWave asks:** *"How much stock do we need for December? Last year we ran out of best-sellers in week 2 and over-ordered everything else."*

**Time:** about 10 hours. **Prerequisites:** Module 01 (Pandas dates, plotting), Module 04 (honest evaluation, leakage), Module 05 (gradient boosting).

## Where you are

Every model so far has predicted one column from others: churn from recency, spend from visits. Forecasting is different. There is one number per week, and the only clue about next week is the weeks before it. The usual tools need rethinking:

- **No shuffled train/test split.** Shuffling leaks the future into training.
- **Features are only allowed if they would be known on the day you forecast.** A rolling average or lag computed carelessly can include the answer.
- **The calendar is not a nuisance.** It is most of the signal.

ShopWave's data makes this concrete. Two years of weekly revenue, from December 2009 to December 2011, have a flat trend and a huge yearly season. November sells about 1.76× an average month, and the shop closes over Christmas. Only two Novembers exist to learn from. That is a realistic, slightly uncomfortable amount of history, and this module is honest about what it allows.

You will build the classical toolkit (baselines, exponential smoothing, ARIMA) and the modern one (gradient boosting on lag features). You will test them all in one rolling backtest and end with a real decision: the December order.

## What you will be able to do afterwards

- Resample, smooth and decompose a time series into trend, seasonality and noise, and read its autocorrelation.
- Build naive, seasonal-naive, mean and drift baselines by hand, and know why seasonal naive is so hard to beat.
- Choose an error metric that suits the business (MAE over MAPE when some weeks are near zero).
- Run a rolling-origin backtest that never peeks at the future.
- Explain and fit exponential smoothing, Holt and Holt-Winters, and know when two years is too little.
- Explain stationarity, differencing, AR and MA, choose a SARIMA from ACF/PACF, check residuals and prediction intervals.
- Turn a series into a supervised-learning table without leakage, and compare direct and recursive multi-step forecasting.
- Deliver a forecast with an honest range and a recommendation a buyer can act on.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 07.1 | [Time series: trend, seasonality, noise, decomposition](01-time-series-basics/) | 1.5 h |
| 07.2 | [Baselines and honest evaluation: time-ordered splits, MAPE, backtesting](02-baselines-and-evaluation/) | 1.5 h |
| 07.3 | [Smoothing and Holt-Winters](03-smoothing-holt-winters/) | 1.5 h |
| 07.4 | [ARIMA and SARIMA](04-arima-sarima/) | 2 h |
| 07.5 | [Forecasting with machine learning: lag features and gradient boosting](05-ml-forecasting/) | 1.5 h |
| 07.P | [Mini-project: the December stock forecast](project/) | 2 h |

## The scoreboard you will build

Every model goes through the same rolling backtest: 8 forecast dates from April to October 2011, 5 weeks ahead each time, scored by mean absolute error in pounds per week.

| model | lesson | mean MAE |
|---|---|---|
| GBM recursive (yesterday, last week, weekday, month) | 07.5 | £30.2k |
| **seasonal naive** (same week last year) | 07.2 | **£32.4k** |
| SARIMA(1,0,1)(0,1,0)₅₂ | 07.4 | £34.6k |
| Holt-Winters (seeded with the first year) | 07.3 | £34.7k |
| GBM direct (lags ≥ 35 days + calendar) | 07.5 | £34.7k |

The headline is honest and important: with two years of history, almost nothing beats "same week last year". The models' real value is the **ranges** they give (07.4) and cross-checking the plan (07.P).

## The maths in this module

Weighted averages (exponential smoothing), differences (this week minus last year), the correlation from 01.8 applied to a series and its own past (autocorrelation), and the absolute and squared errors from 02.3. Every formula is computed by hand on 5 to 12 numbers before it is used on real data.

## Data

Both sheets of Online Retail II (December 2009 – December 2011), loaded with `load_retail()` and cleaned with the usual `clean_retail()`.

- **Overlap.** The first nine days of December 2010, which appear in both sheets, are counted once.
- **One-off orders.** Two single orders of over 70,000 units, each cancelled the same day, are removed.
- **End date.** The series stops at the last complete week, Sunday 4 December 2011. The raw data ends mid-week on 9 December, and a partial week (or a four-day "December") would look like a collapse in sales.
