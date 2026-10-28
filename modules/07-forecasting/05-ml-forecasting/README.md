# 07.5 · Forecasting with machine learning

⏱ about 1.5 hours · Prerequisites: 07.2, Module 05 (gradient boosting), 05.8 (permutation importance) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The head of data has a reasonable question: *"You spent Module 05 learning gradient boosting, the model that wins on business tables. Why are we forecasting with methods from the 1960s?"*

Gradient boosting does not understand "time". It understands tables: rows, feature columns, a target. To use it for forecasting you have to **turn the series into a table** yourself, deciding what the model may know about each day before that day happens. Done carelessly, this is the easiest place in all of machine learning to leak the future into your features. Done well, it lets the model use things the classical methods cannot: the weekday, the month, recent trading, and later prices, promotions and thousands of products at once.

## 2. The idea

A shop manager forecasting tomorrow does not run SARIMA in their head. They think: *"Tomorrow is a Thursday, Thursdays are busy; we're in November, so it's peak season; and this week has been running a bit hot."*

Each of those is a **feature**: day of the week, month, recent level. Write them down as columns, one row per day, with the actual revenue as the target, and you have an ordinary regression table. The only rule is the one from 07.1, now in a stricter form: **every feature must be known at the moment you make the forecast.**

## 3. Definition

- **Lag feature**: the value *k* steps earlier, e.g. `lag_7` = same weekday last week.
- **Rolling feature**: a summary (mean, max) of a window of *past* values, shifted so it ends before the row's own date.
- **Calendar features**: day of week, month, day of year, days to Christmas, holiday flags.
- **Direct multi-step forecasting**: forecast every step of the horizon from information available at the origin, using only lags at least as long as the horizon (or one model per step).
- **Recursive multi-step forecasting**: train a one-step-ahead model, forecast the next step, feed the forecast back in as the newest lag, repeat.
- **TimeSeriesSplit**: cross-validation in which every fold trains on the past and tests on the block after it.
- **Global model**: one model trained across many related series (e.g. every product), with an identifier column.

## 4. Formula

The forecasting problem becomes ordinary supervised learning:

$$\hat y_t = f\big(y_{t-k_1},\ y_{t-k_2},\ \dots,\ \tfrac{1}{w}\textstyle\sum_{i=1}^{w} y_{t-L-i},\ \text{calendar}(t)\big)$$

where $f$ is any regression model (here, gradient boosting from 05.3–05.4).

**Validity rule for direct forecasting.** To forecast up to $H$ steps ahead from origin $T$:

$$\text{every lag } k \ge H \quad\text{(here } H = 35 \text{ days, so lags} \ge 35\text{)}$$

**Recursive forecasting** replaces unknown lags with earlier forecasts:

$$\hat y_{T+1} = f(y_T, y_{T-6}, \dots) \qquad \hat y_{T+2} = f(\hat y_{T+1}, y_{T-5}, \dots) \qquad \dots$$

## 5. By hand

Twelve days of revenue (£k), starting on a Monday:

| | Mon 7 | Tue 8 | Wed 9 | Thu 10 | Fri 11 | Sat 12 | Sun 13 | Mon 14 | Tue 15 | Wed 16 | Thu 17 | Fri 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| y | 30 | 32 | 28 | 40 | 25 | 0 | 18 | 31 | 33 | 29 | 42 | 26 |

Build the table for the last five days. `lag_1` = yesterday, `lag_7` = same weekday last week, `mean_3` = average of the **three days before** (not including the day itself), `dow` = day of week (0 = Monday).

| day | lag_1 | lag_7 | mean_3 | dow | **y** |
|---|---|---|---|---|---|
| Mon 14 | 18 | 30 | (25 + 0 + 18) / 3 = **14.33** | 0 | **31** |
| Tue 15 | 31 | 32 | (0 + 18 + 31) / 3 = **16.33** | 1 | **33** |
| Wed 16 | 33 | 28 | (18 + 31 + 33) / 3 = **27.33** | 2 | **29** |
| Thu 17 | 29 | 40 | (31 + 33 + 29) / 3 = **31.00** | 3 | **42** |
| Fri 18 | 42 | 25 | (33 + 29 + 42) / 3 = **34.67** | 4 | **26** |

Notice `lag_7` is nearly always closer to `y` than `lag_1`: Thursday looks like last Thursday, not like Wednesday. A model would learn that from the rows.

**The leak to avoid.** If `mean_3` for Thursday 17 included Thursday itself, it would be (33 + 29 + 42) / 3 = 34.67, which contains the answer 42. In Pandas the difference is one call: `y.shift(1).rolling(3).mean()` (correct) vs `y.rolling(3).mean()` (leaks).

**Direct vs recursive.** To forecast Friday 18 from Monday 14, `lag_1` (Thursday 17) is not yet known. Direct forecasting refuses to use it and sticks to lags of 5 days or more. Recursive forecasting predicts Tuesday, Wednesday and Thursday first and uses its own Thursday forecast as `lag_1`.

## 6. In code

```python
def daily_table(d):
    t = pd.DataFrame({"y": d})
    t["lag_35"] = d.shift(35)                          # known 35 days ahead
    t["mean_28_lag_35"] = d.shift(35).rolling(28).mean()
    t["dow"] = d.index.day_of_week
    t["month"] = d.index.month
    return t

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

for tr, te in TimeSeriesSplit(n_splits=5, test_size=60).split(rows):
    model.fit(rows.iloc[tr][features], rows.iloc[tr]["y"])
```

The notebook wraps a direct model (`gbm`) and a recursive one (`recursive_gbm`, trained on `lag_1`, `lag_7`, `dow` and `month`, forecasting day by day). Each predicts daily revenue, forces Saturdays to zero, and sums to weekly totals so it can enter the same backtest as every other model.

## 7. On real data

**The table.** 672 usable daily rows, with features `lag_35`, `mean_28_lag_35`, `dow`, `month`, `day_of_year` and `days_to_xmas`.

**TimeSeriesSplit** (5 folds of 60 days) gives a daily MAE of £5.7k–£10.3k. The worst fold is the September–October ramp-up.

**What the model uses** (permutation importance on September–December 2011): **day of week** dominates (about £9.6k a day of extra error when shuffled), and `lag_35` adds a little (£0.5k). `day_of_year` and `days_to_xmas` are useless or slightly harmful on new data. With one previous autumn, they let the trees memorise particular 2010 dates.

**Backtest** (same 8 origins and 5-week horizon):

| model | mean MAE |
|---|---|
| **GBM recursive** (lag 1, lag 7, weekday, month) | **£30.2k** |
| seasonal naive | £32.4k |
| SARIMA | £34.6k |
| Holt-Winters | £34.7k |
| GBM direct (lags ≥ 35 days + calendar) | £34.7k |

- **Recursive is the first model in this module to beat seasonal naive on average.** It is ahead by about £2k a week, roughly 7%. It uses last week's actual trading level, which seasonal naive ignores. Over only eight origins, treat this as suggestive, not proof.
- **The direct model improves without the date features.** Removing `day_of_year` and `days_to_xmas` takes it from £34.7k to **£31.4k** (exercise 2). More features is not automatically better.
- **The year-on-year trap.** Adding "same day last year" (`lag_364`) makes the direct model much *worse*: £49.7k. Only 370 rows have that feature, all from 2011, so the model never sees a November target and cannot forecast one. Trees cannot extrapolate.
- **Weekly is too thin.** A weekly table (exercise 3) has 18 usable rows at the first origin and scores £36.6k.

**Products.** The buyers order units, not revenue. The five best-selling products each have their own spikes and quiet spells. The industry answer is one *global* gradient-boosting model across all products; that is the project's stretch task.

## 8. Where it breaks

- **Leakage is easy.** A rolling mean without `shift`, a lag shorter than the horizon, or a feature computed on the full series (like a yearly average) all leak the future. Check every feature against the question: "would I know this on the origin date?"
- **Trees cannot extrapolate.** They predict values within the range of training targets. A new record peak, steady growth, or a target range missing from training (the `lag_364` trap) all defeat them. Classical models extrapolate trends; trees do not.
- **Recursive errors compound.** Here fresher data outweighed that, but on longer horizons or noisier series a recursive model can drift far off.
- **Date features overfit short histories.** "Day of year" with one year of history is a lookup table of last year's accidents.
- **One series is ML's weakest case.** The approach shines with many related series and extra drivers (price, promotions, weather, traffic), none of which this module has.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: build the row for Saturday 12 November (`lag_1`, `lag_7`, `mean_3`, `dow`). Check in Pandas.
2. Remove `days_to_xmas` and `day_of_year` and backtest again. Better or worse? Why?
3. Build a weekly table (`lag_5`, `lag_52`, `week_of_year`) and backtest gradient boosting on it.
4. Train a 90th-percentile (quantile) model and forecast 6 November – 4 December 2011. How many actual weeks fall below it?

**Quiz**

1. How do you turn a time series into a supervised-learning table?
2. Why must the rolling mean be computed on `y.shift(1)` rather than `y`?
3. What is the difference between direct and recursive multi-step forecasting?
4. Why does the direct model here use only lags of 35 days or more?
5. Why did adding `lag_364` make the model worse?

## 10. Recap

- Any regression model can forecast once the series is a table of lags, rolling means and calendar features, as long as every feature is known at the origin.
- `TimeSeriesSplit` and the rolling backtest are the only honest ways to validate.
- On ShopWave, a recursive gradient-boosting model (yesterday, last week, weekday, month) is the first to beat seasonal naive: £30.2k vs £32.4k a week.
- Date-identifying features and a 364-day lag both hurt: with one previous year, trees memorise accidents and cannot extrapolate to unseen ranges.
- ML forecasting pays off most with many related series and extra drivers; the project's stretch task points that way.

**Next:** [07.P Mini-project: the December stock forecast](../project/) brings every model together on the real decision.
