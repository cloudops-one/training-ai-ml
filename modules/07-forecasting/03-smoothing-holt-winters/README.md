# 07.3 · Smoothing and Holt-Winters

⏱ about 1.5 hours · Prerequisites: 07.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Seasonal naive won the backtest in 07.2, but the buyers are uneasy about it. *"So the plan is to order exactly what we sold last year? What if this year is running hotter? What about the slow drift we keep hearing about?"*

They have a point. Seasonal naive uses exactly one number for each forecast, the same week last year, and ignores everything since. It cannot notice a level change or a trend.

Exponential smoothing fixes that. It keeps a running estimate of the level, updated every week, with recent weeks counting more than old ones. Holt adds a trend, and Holt-Winters adds a season. These have been the workhorse forecasting methods in retail for sixty years, and it is fair to ask whether they beat seasonal naive.

## 2. The idea

You are estimating how long your commute takes. You could average every commute you have ever made (the mean), or trust yesterday's alone (naive). Most people do something in between: *"usually about 35 minutes, but this week it has been more like 40, so I'll plan for 38."*

That is exponential smoothing. You keep a running estimate (the **level**), and after each new observation you nudge it part of the way towards what you just saw. How far you nudge it is a single dial, **α**:

- α = 1: jump all the way to the latest value (naive).
- α = 0: never move (stuck at the starting value).
- α = 0.3: move 30% of the way, so recent weeks matter most and old weeks fade out *exponentially*.

Holt adds a second running estimate, the **trend** (how much the level changes per week), with its own dial β. Holt-Winters adds a third, the **seasonal** effect for each week of the year, with dial γ.

## 3. Definition

- **Simple exponential smoothing (SES)**: forecast = current level; the level is updated as a weighted average of the new observation and the old level.
- **Smoothing parameter α** (0–1): how quickly the level reacts.
- **Holt's linear method**: SES plus a trend term with smoothing parameter β.
- **Damped trend**: the trend is multiplied by φ (0–1) at each future step, so it fades instead of extrapolating forever.
- **Holt-Winters**: Holt plus a seasonal term with smoothing parameter γ and season length *m* (52 for weekly).
- **Initialisation**: the starting level, trend and seasonal values. With a 52-week season this matters a lot, because there are 52 starting seasonal values to choose.

## 4. Formula

**SES.** Level $\ell_t$, smoothing parameter $\alpha$:

$$\ell_t = \alpha\, y_t + (1-\alpha)\,\ell_{t-1} \qquad\qquad \hat y_{t+h} = \ell_t$$

Unrolled, the level is a weighted average of all past values with weights $\alpha,\ \alpha(1-\alpha),\ \alpha(1-\alpha)^2, \dots$, which shrink exponentially. Hence the name.

**Holt (damped).** Adds trend $b_t$ and damping $\phi$:

$$\ell_t = \alpha\, y_t + (1-\alpha)(\ell_{t-1} + \phi\, b_{t-1}) \qquad b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)\,\phi\, b_{t-1}$$

$$\hat y_{t+h} = \ell_t + (\phi + \phi^2 + \dots + \phi^h)\, b_t$$

**Holt-Winters (additive).** Adds the seasonal term $s_t$ with period $m$:

$$\ell_t = \alpha (y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + \phi b_{t-1}) \qquad s_t = \gamma (y_t - \ell_t) + (1-\gamma)\, s_{t-m}$$

$$\hat y_{t+h} = \ell_t + (\phi + \dots + \phi^h)\, b_t + s_{t+h-m}$$

In words: remove last year's seasonal effect before updating the level, then update this week's seasonal effect with whatever the level could not explain.

## 5. By hand

Five weeks of revenue (£k): **100, 120, 110, 130, 125**. Start the level at the first value, ℓ = 100, and use **α = 0.5**.

| week | actual | forecast (= old level) | error | new level = 0.5 × actual + 0.5 × old level |
|---|---|---|---|---|
| 2 | 120 | 100 | 20 | 0.5 × 120 + 0.5 × 100 = **110** |
| 3 | 110 | 110 | 0 | 0.5 × 110 + 0.5 × 110 = **110** |
| 4 | 130 | 110 | 20 | 0.5 × 130 + 0.5 × 110 = **120** |
| 5 | 125 | 120 | 5 | 0.5 × 125 + 0.5 × 120 = **122.5** |

Forecast for week 6 (and every week after, since SES has no trend or season): **122.5**.

Now **α = 0.2**, which is more cautious:

| week | actual | forecast | error | new level = 0.2 × actual + 0.8 × old level |
|---|---|---|---|---|
| 2 | 120 | 100 | 20 | 24 + 80 = **104** |
| 3 | 110 | 104 | 6 | 22 + 83.2 = **105.2** |
| 4 | 130 | 105.2 | 24.8 | 26 + 84.16 = **110.16** |
| 5 | 125 | 110.16 | 14.84 | 25 + 88.13 = **113.13** |

Small α means a smooth, slow forecast that lags behind real changes. Large α means a jumpy forecast that chases noise. Fitting picks the α that would have made the smallest one-step errors on the history.

## 6. In code

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

ses  = SimpleExpSmoothing(train, initialization_method="estimated").fit()
holt = ExponentialSmoothing(train, trend="add", damped_trend=True,
                            initialization_method="estimated").fit()

first = train.iloc[:52]                    # seed the season with the first year
hw = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add",
                          seasonal_periods=52, initialization_method="known",
                          initial_level=first.mean(), initial_trend=0.0,
                          initial_seasonal=(first - first.mean()).values).fit()
hw.forecast(5); hw.params
```

The notebook first fixes α = 0.5 with `optimized=False` and checks that statsmodels reproduces the by-hand table exactly (fitted values 100, 100, 110, 110, 120; forecast 122.5).

## 7. On real data

**Fitted parameters** (weekly revenue up to 16 October 2011):

- **SES**: α = 0.586.
- **Holt**: α = 0.534, β = **0.000**, φ = 0.80. Holt finds no lasting trend, which matches the flat trend in 07.1.
- **Holt-Winters**: its default initialisation fails with *"less than two full seasonal cycles"*, because only 98 weeks are available at the last origin and 104 are needed. Seeding it with the first year instead works. The fitted α, β and γ then all come out at **0**: with one year after the seed, the best Holt-Winters can do is keep the first-year pattern unchanged, which is essentially seasonal naive.

**Backtest** (same 8 origins and 5-week horizon as 07.2):

| model | mean MAE |
|---|---|
| **seasonal naive** | **£32.4k** |
| Holt-Winters | £34.7k |
| SES | £37.4k |
| Holt (damped) | £38.1k |

Nothing beats seasonal naive on average. SES and Holt win the quiet summer origins (26 June, 24 July), where "like the last few weeks" is right. They lose badly whenever the season turns. The worst case is the 1 May origin, straight after a bank-holiday week.

**Monthly is friendlier** (exercise 4). There are 12 seasonal values instead of 52, no zero months, and multiplicative seasonality is allowed. Seeded the same way, a monthly multiplicative Holt-Winters forecasts November 2011 at £1.22m against an actual £1.16m (+5.6%). It cannot say *which week* the peak arrives, though.

## 8. Where it breaks

- **Short history with a long season.** Fifty-two seasonal values cannot be learned from two years. Holt-Winters collapses into "repeat last year". It needs three or more seasons to add value.
- **Zeros rule out multiplicative seasonality.** ShopWave's closed Christmas weeks force the additive form on weekly data.
- **α is fitted on squared error.** It is pulled by the biggest jumps (exercise 2: best MAE at α = 0.3, chosen α = 0.59). If your business cares about typical error, check the fit with MAE too.
- **Trends are dangerous to extrapolate.** Use damping by default. On ShopWave it makes no difference, because there is no trend to damp.
- **One series at a time.** Exponential smoothing knows nothing about promotions, prices or holidays unless you build them in. Lesson 07.5 shows how machine learning can use such extra information.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: SES with α = 0.3, starting level 50, on `[50, 60, 55, 70]`. What is the forecast for the fifth value?
2. Fit SES with α fixed at 0.1 … 0.9 and compare in-sample MAE. Which α is best, and does it match the one statsmodels chooses?
3. Backtest Holt without damping. Better or worse, and why?
4. Fit a monthly multiplicative Holt-Winters and forecast November 2011.

**Quiz**

1. What does α control? What happens at α = 1 and at α = 0?
2. The level is 120, α = 0.3, and this week sold 140. What is the new level?
3. Why damp the trend?
4. Why does statsmodels' default Holt-Winters initialisation fail on ShopWave's weekly data?
5. What does it mean that the fitted α, β and γ are all near zero?

## 10. Recap

- SES keeps a running level and nudges it by α towards each new value. Old weeks fade exponentially.
- Holt adds a (preferably damped) trend; Holt-Winters adds one seasonal value per period.
- With only two years, weekly Holt-Winters has nothing to learn beyond the first year and behaves like seasonal naive.
- In the backtest nothing beats seasonal naive's £32.4k. SES and Holt win quiet summer weeks but miss every seasonal turn.
- Monthly data suits Holt-Winters better (forecast within 6% of November 2011), at the cost of weekly detail.

**Next:** [07.4 ARIMA and SARIMA](../04-arima-sarima/) attacks the same problem from another angle: model what is left after removing the season.
