# 07.4 · ARIMA and SARIMA

⏱ about 2 hours · Prerequisites: 07.1–07.3, correlation (01.8) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The finance director has read that ARIMA is "the statistical standard for forecasting" and wants to know why you are using something as simple as "last year's week". They also want something the baselines cannot give: *"Don't just give me a number. Tell me how wrong it could be."*

ARIMA is the other great classical family next to exponential smoothing. It models a series by how each value depends on **recent values** and **recent surprises**, after removing trends and seasons by differencing. Its seasonal version, SARIMA, handles ShopWave's yearly pattern. And because it is a proper statistical model, it comes with **prediction intervals**: a range for every forecast week.

## 2. The idea

Picture a thermostat-controlled room. The temperature is not random. If it was warm a minute ago, it is probably still warm now, but drifting back towards the setpoint. That is **auto-regression (AR)**: today depends on yesterday.

Now picture a shop that gets a surprise coach party on Monday. Tuesday is also a bit busier (word of mouth, returns), but by Thursday the effect has gone. That is a **moving average (MA)** effect: today depends on yesterday's *surprise*.

Both ideas assume the series has a stable average to drift around. ShopWave's revenue swings with the calendar, so first we remove the calendar by **differencing**. We look at "this week minus the same week last year" instead of raw revenue. That is the **I** (integrated) and the **S** (seasonal) in SARIMA.

## 3. Definition

- **Stationary series**: its mean, variance and autocorrelation do not depend on when you look. No trend and no calendar pattern.
- **Differencing**: replace each value with its change from an earlier value. First difference: $y_t - y_{t-1}$. Seasonal difference: $y_t - y_{t-m}$.
- **ADF test** (Augmented Dickey-Fuller): tests for a *unit root*, a level that wanders without returning. p < 0.05 → no unit root. It does **not** test for seasonality.
- **AR(p)**: the value depends on the previous *p* values.
- **MA(q)**: the value depends on the previous *q* forecast errors (shocks).
- **ARIMA(p, d, q)**: difference *d* times, then fit AR(p) + MA(q).
- **SARIMA(p, d, q)(P, D, Q)ₘ**: adds a seasonal part with period *m*: *D* seasonal differences and seasonal AR/MA terms at lags *m*, 2*m*, …
- **ACF / PACF**: autocorrelation, and *partial* autocorrelation (correlation at lag *k* after removing lags 1 … *k*−1). Used to choose *p* and *q*.
- **Ljung-Box test**: checks whether residuals still contain autocorrelation. A large p-value means they look like noise.
- **Prediction interval**: a range that should contain the actual value with a stated probability (e.g. 95%).

## 4. Formula

**Differencing.**

$$\Delta y_t = y_t - y_{t-1} \qquad\qquad \Delta_{52}\, y_t = y_t - y_{t-52}$$

**AR(1)** with constant $c$, coefficient $\phi$ and noise $\varepsilon_t$:

$$y_t = c + \phi\, y_{t-1} + \varepsilon_t \qquad \text{long-run mean} = \frac{c}{1-\phi}\ \ (|\phi| < 1)$$

**MA(1)** with mean $\mu$ and coefficient $\theta$:

$$y_t = \mu + \varepsilon_t + \theta\, \varepsilon_{t-1}$$

**ARMA(1,1)** combines them. **SARIMA(1,0,1)(0,1,0)₅₂** applies ARMA(1,1) to the seasonal difference $w_t = y_t - y_{t-52}$:

$$w_t = c + \phi\, w_{t-1} + \varepsilon_t + \theta\, \varepsilon_{t-1}$$

**Reading ACF/PACF:**

| pattern | ACF | PACF |
|---|---|---|
| AR(p) | decays gradually | cuts off after lag *p* |
| MA(q) | cuts off after lag *q* | decays gradually |
| nothing left | all inside the band | all inside the band |

## 5. By hand

**Differencing** six weeks (£k): 100, 104, 110, 108, 115, 121.

| week | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| y | 100 | 104 | 110 | 108 | 115 | 121 |
| Δy | — | 4 | 6 | −2 | 7 | 6 |

The original climbs steadily; the differences hover around 4. One difference turned "trending" into "stable mean", so *d* = 1 would be right for this series.

**AR(1)** with c = 20, φ = 0.8, starting at y = 50 (noise set to zero to see the mechanism):

- step 1: 20 + 0.8 × 50 = **60**
- step 2: 20 + 0.8 × 60 = **68**
- step 3: 20 + 0.8 × 68 = **74.4**
- long-run mean: 20 / (1 − 0.8) = **100**

The series climbs towards 100, closing 20% of the gap each step. An AR process remembers *values*.

**MA(1)** with μ = 100, θ = 0.5, and last week's surprise ε = +10:

- 1 step ahead: 100 + 0.5 × 10 = **105** (next week's surprise is unknown, so expect 0)
- 2 steps ahead: **100** (the surprise has been forgotten)

An MA process remembers *shocks*, and only for *q* steps.

## 6. In code

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

adfuller(weekly)[1]                         # p-value
sd = weekly.diff(52).dropna()               # this year minus last year
plot_acf(sd, lags=15); plot_pacf(sd, lags=15)

fit = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 1, 0, 52), trend="c").fit(disp=False)
fit.summary()
fc = fit.get_forecast(steps=5)
fc.predicted_mean, fc.conf_int()           # forecast and 95% interval
```

statsmodels prints optimiser notices ("non-stationary starting parameters") on short series. The notebook hides those specific notices so the output stays readable, and says so; the residual check and the backtest are what tell you whether the model is any good.

## 7. On real data

**Stationarity.** Raw weekly revenue already passes the ADF test (p = 0.0054), because its level does not wander: the trend is flat. But ADF ignores seasonality, which is obviously there. So we take the 52-week difference (ADF p < 0.0001). That leaves 53 weeks of "this year minus last year".

**ACF/PACF of the seasonal difference.** Every bar sits inside the ±0.27 band; the largest is lag 1 at −0.14. Nothing is left to model once last year is subtracted. That points at SARIMA(0,0,0)(0,1,0)₅₂ with a constant: "same week last year plus the average year-on-year change".

**Fitting more anyway.** We fit SARIMA(1,0,1)(0,1,0)₅₂ with a constant on data up to 16 October 2011. φ and θ come out at **essentially zero**, with standard errors in the hundreds. The data refused the extra terms, and the fitted model is in effect "last year's week minus about £3.8k". The residuals pass the Ljung-Box test (p = 0.57, 0.89 and 0.80 at lags 4, 8 and 12), so no pattern is left.

**Prediction intervals.** From 16 October, each 95% interval is roughly ±£77k around the forecast, and the actual weeks fall inside them. Across all 40 backtest weeks, though, only **85%** (34 of 40) land inside a "95%" interval. The intervals are too narrow, mostly around bank holidays and the September ramp-up, so widen them before planning stock on them.

**Backtest:**

| model | mean MAE |
|---|---|
| **seasonal naive** | **£32.4k** |
| SARIMA(1,0,1)(0,1,0)₅₂ | £34.6k |
| Holt-Winters | £34.7k |

The finance director gets an honest answer: on two years of data, SARIMA is not more accurate than last year's week. Its real contribution is the **range**, once calibrated.

## 8. Where it breaks

- **Short history.** Seasonal differencing at lag 52 throws away the first year. With 105 weeks, 53 remain, too few to estimate anything subtle. SARIMA needs several seasons to earn its keep.
- **Weekly seasonality is awkward.** A period of 52 is slow to fit and fragile. Years have 52.18 weeks, so the "same week" drifts by a day a year. Monthly data (exercise 4) is far easier.
- **Calendar effects are not built in.** Easter and bank holidays move; SARIMA does not know. Add them as extra regressors (the X in SARIMAX) or accept larger errors around them.
- **Intervals assume the model is right.** They ignore uncertainty about the parameters and the model choice, so they are usually too narrow. Check their coverage in a backtest, as exercise 3 does.
- **ADF can mislead.** Passing ADF does not mean "no seasonality", and failing it on a short series can be noise. Look at the plots too.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: difference `[200, 210, 205, 220, 230]`; run AR(1) with c = 30, φ = 0.7 from y = 60 for three steps.
2. Backtest SARIMA with orders (1,0,0), (0,0,1) and (0,0,0). Which is best, and what is (0,0,0)(0,1,0,52) in plain words?
3. What fraction of actual weeks falls inside SARIMA's 95% interval across the backtest?
4. Fit a monthly SARIMA(0,0,0)(0,1,0,12) and forecast November 2011.

**Quiz**

1. What does "stationary" mean, and why does ARMA need it?
2. Difference `[50, 53, 57, 56]` once.
3. AR(1) with c = 10, φ = 0.5, last value 40: what is the next forecast?
4. The ACF decays slowly and the PACF cuts off after lag 2. Which model does that suggest?
5. Spell out every number in SARIMA(1,0,1)(0,1,0)₅₂.

## 10. Recap

- ARIMA models what is left after differencing: AR terms remember values, MA terms remember shocks.
- Seasonal differencing (this week minus the same week last year) removes ShopWave's yearly pattern; ACF/PACF of what is left guide *p* and *q*.
- On ShopWave nothing is left after seasonal differencing: φ and θ fit to zero, and SARIMA becomes "last year's week plus a constant".
- SARIMA's backtest MAE (£34.6k) does not beat seasonal naive (£32.4k). Its value is the prediction interval, which should be checked: only 85% coverage for a "95%" interval.
- Always check residuals (Ljung-Box) and interval coverage, not just the point forecast.

**Next:** [07.5 Forecasting with machine learning](../05-ml-forecasting/) turns the series into a table and lets gradient boosting use calendar features the classical models cannot.
