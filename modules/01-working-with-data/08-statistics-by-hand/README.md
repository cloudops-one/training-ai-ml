# 01.8 · Statistics you need: mean, median, spread, percentiles, correlation

⏱ about 2 hours · Prerequisites: 01.7 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The CEO asks: *"What does a typical ShopWave order look like?"* You answer "the average order is £480". Operations laughs: *"Half our orders are under £300. Your average is three wholesalers."*

Both of you are right, and that is the problem. "Typical" needs a precise definition, and so do "spread", "unusually large" and "related to". Those definitions are the handful of statistics in this lesson. They are the vocabulary every chart from 01.7 was drawn in, and every model from Module 02 onwards is built from.

**Why this matters for ML:** regression (Module 02) literally *is* means and correlations. Scaling features (Module 04) uses mean and standard deviation. Outlier removal uses percentiles and IQR. Evaluation metrics are averages of errors. If you can compute these six numbers by hand, nothing in the next ten modules is a black box.

## 2. The idea

Six numbers describe a column well enough for most decisions:

| Question | Statistic |
|---|---|
| Where is the centre? | **mean** (balance point), **median** (middle value) |
| How spread out is it? | **standard deviation** (typical distance from the mean) |
| What are the landmarks? | **percentiles** (25th, 50th, 75th) and the **IQR** |
| What counts as unusual? | beyond **1.5 × IQR** from the box |
| Do two columns move together? | **correlation** (−1 to +1) |

The mean is the balance point of a seesaw: one heavy child far out (a £10,000 order) tips it. The median does not care how heavy that child is, only that they are on one side. That is the whole CEO argument.

## 3. Definition

For $n$ values $x_1, \dots, x_n$:

- **Mean** $\bar{x}$: the sum divided by the count.
- **Median**: sort the values; the middle one (or the mean of the two middle ones if $n$ is even).
- **Variance** $s^2$: the average squared distance from the mean. **Standard deviation** $s$: its square root, so it is back in the original units.
- **Percentile** $P_k$: the value below which $k\%$ of the data lies. $P_{25}, P_{50}, P_{75}$ are the **quartiles** $Q_1, Q_2, Q_3$. $Q_2$ is the median.
- **IQR**: $Q_3 - Q_1$, the width of the middle half.
- **Outlier (Tukey's rule)**: any value below $Q_1 - 1.5 \times \text{IQR}$ or above $Q_3 + 1.5 \times \text{IQR}$. This is what the box plot draws as dots.
- **Correlation** $r$ (Pearson): how closely two columns follow a straight line together. $+1$ perfect uphill, $-1$ perfect downhill, $0$ no straight-line relationship.

## 4. Formula

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\qquad\qquad
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\;\sqrt{\sum (y_i - \bar{y})^2}}$$

Reading them:

- $s$: take each distance from the mean, square it (so negatives do not cancel and big distances count more), average, square-root to undo the squaring. The $n-1$ instead of $n$ is a small correction for the fact that we estimated $\bar{x}$ from the same data; Pandas uses $n-1$ by default, NumPy uses $n$. On big data the difference vanishes.
- $r$: the numerator is the same "do they move together" sum you met in linear regression (02.2). The denominator rescales it so the answer is always between −1 and +1, regardless of units.

## 5. By hand

**Six order values:** 10, 20, 20, 30, 40, 60 (pounds).

- **Mean**: $(10+20+20+30+40+60)/6 = 180/6 = 30$.
- **Median**: sorted already; $n = 6$ is even, middle two are 20 and 30, median $= 25$. Already lower than the mean: the £60 order pulls the mean up.
- **Standard deviation**:

| $x$ | $x - \bar{x}$ | $(x-\bar{x})^2$ |
|---|---|---|
| 10 | −20 | 400 |
| 20 | −10 | 100 |
| 20 | −10 | 100 |
| 30 | 0 | 0 |
| 40 | 10 | 100 |
| 60 | 30 | 900 |
| | **sum** | **1600** |

$s^2 = 1600 / (6-1) = 320$, $s = \sqrt{320} \approx 17.9$. A typical order sits about £18 from the £30 mean.

- **Quartiles** (simple method: medians of the halves): lower half 10, 20, 20 → $Q_1 = 20$; upper half 30, 40, 60 → $Q_3 = 40$. $\text{IQR} = 20$.
- **Outlier fences**: $20 - 30 = -10$ and $40 + 30 = 70$. Nothing is outside, so no dots on the box plot. Now add one order of **£150**: $Q_3$ and the fence barely move, but 150 > 70, so it is flagged. The mean jumps from 30 to 47; the median only from 25 to 30. *That* is the CEO conversation in numbers.

**Correlation**, using the five customers from lesson 02.2 (visits $x$, spend $y$):

From that lesson: $\sum (x-\bar{x})(y-\bar{y}) = 500$, $\sum (x-\bar{x})^2 = 40$. One more column: $\sum (y-\bar{y})^2 = 2500+400+100+900+2500 = 6400$.

$$r = \frac{500}{\sqrt{40}\sqrt{6400}} = \frac{500}{6.32 \times 80} = \frac{500}{505.96} \approx 0.988$$

Visits and spend are almost perfectly linearly related in that toy table. On real customers it will be lower.

## 6. In code

```python
s = pd.Series([10, 20, 20, 30, 40, 60])
s.mean(), s.median(), s.std()                    # 30, 25, 17.89
s.quantile([.25, .5, .75])                       # quartiles
q1, q3 = s.quantile(.25), s.quantile(.75)
iqr = q3 - q1
outliers = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]
s.describe()                                     # all of the above at once
customers["orders"].corr(customers["spend"])     # Pearson r
customers.corr(numeric_only=True)                # the matrix behind the heatmap
```

Note: Pandas' `quantile` uses linear interpolation between sorted values, which on tiny samples can differ slightly from the "median of halves" method you used on paper. On real data they agree to many decimals.

## 7. On real data

The notebook computes all six statistics for ShopWave order values and for the customer table, shows that the mean order (about £480) is far above the median (about £300), flags the outlier orders with Tukey's rule, and computes the correlation matrix you saw as a heatmap in 01.7. It ends by checking that `describe()` agrees with your hand formulas.

## 8. Where it breaks

- **Mean on skewed data.** ShopWave's order values, customer spend, product quantities: all right-skewed. Report the median, or the mean *and* the median together.
- **Standard deviation on skewed data.** $s$ assumes a roughly symmetric spread. On skewed data "mean ± 2 s" can reach below zero for a quantity that cannot be negative. Use percentiles instead.
- **Tukey's rule is a flag, not a verdict.** 1.5 × IQR flags thousands of real wholesale orders. Investigate; do not delete.
- **Correlation measures straight lines only.** A perfect U-shape has $r \approx 0$. Always look at the scatter.
- **Correlation is not causation.** Ice-cream sales correlate with drownings. Both follow summer.
- **One outlier can swing $r$.** A single giant customer can turn $r = 0.3$ into $r = 0.8$. Compute it with and without.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: mean, median, standard deviation, quartiles and IQR of 5, 7, 8, 12, 14, 15, 40. Which value is an outlier by Tukey's rule? Confirm in code.
2. On the customer table, compute `spend` mean and median per country for the top 6 countries. Where is the gap biggest and why?
3. Compute the correlation between `orders` and `spend` with all customers, then with the top 1% of spenders removed. How much does $r$ change?
4. Write your own `std()` using the formula with $n-1$ and check it against `Series.std()` on 1,000 random order values.

**Quiz**

1. When do mean and median differ a lot, and which should you report?
2. Why is the standard deviation in the same units as the data, while the variance is not?
3. What does an IQR of £200 tell you?
4. $r = -0.9$ between `days_since_last_order` and `spend_next_month`. In words?
5. Name a pattern between two variables that correlation would miss entirely.

## 10. Recap

- **Mean** = balance point, sensitive to outliers. **Median** = middle value, robust. Report both on skewed data.
- **Standard deviation** = typical distance from the mean, computed from squared deviations.
- **Quartiles and IQR** describe the middle half; **1.5 × IQR** beyond the box flags outliers.
- **Correlation** = "do they move together along a line", from −1 to +1; not cause, not curves.
- Every one of these is a line in `describe()` and `corr()`, and every one reappears inside the models ahead.

**Next:** [01.9 The EDA checklist](../09-eda-checklist/): putting charts and statistics into a repeatable routine.
