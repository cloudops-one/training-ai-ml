# 01.7 · Plotting with Matplotlib and Seaborn: the six charts you actually need

⏱ about 2 hours · Prerequisites: 01.6 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You send the CEO the monthly revenue table from 01.6. The reply: *"Great. Is November good or bad? Are we growing? Which country should we open a warehouse in? I don't have time to read numbers."*

Neither will you, when you have 40 columns and 400,000 rows. Humans find patterns in shapes far faster than in digits. A chart is not decoration; it is how you *see* trend, spread, outliers and relationships, and it is how you catch the mistake a table hides.

**Why this matters for ML:** before any model, you must know the target's distribution (skewed? imbalanced?), which features relate to it (scatter), where the outliers are (box plot) and which features duplicate each other (heatmap). Every one of these is a chart from this lesson. Later modules say "plot it" constantly; this is where you learn how.

## 2. The idea

There are dozens of chart types. You need six, and each answers one question:

| Question | Chart | Example |
|---|---|---|
| How does something change over time? | **Line** | revenue by month |
| How do categories compare? | **Bar** | revenue by country |
| How is one number distributed? | **Histogram** | order values |
| Where is the middle, the spread, the outliers? | **Box plot** | order value per country |
| How do two numbers relate? | **Scatter** | orders vs spend per customer |
| How do many numbers relate at once? | **Heatmap** | correlation between features |

Pick the question, then the chart. Never the other way round.

## 3. Definition

- **Matplotlib**: the base plotting library. Every chart is a `Figure` containing one or more `Axes`. You set titles, labels and limits on the axes.
- **Seaborn**: a layer on top of Matplotlib with better defaults and one-line statistical charts (`histplot`, `boxplot`, `scatterplot`, `heatmap`). Use it for the chart; use Matplotlib to polish it.
- **Histogram**: split a number's range into bins, count how many values fall in each bin, draw bars. Shows the *shape* of the distribution: symmetric, skewed, two humps.
- **Box plot**: a box from the 25th to the 75th percentile, a line at the median, whiskers to the last value within 1.5 × IQR, dots beyond that (outliers). Five numbers that summarise a distribution; see 01.8 for the formulas.
- **Scatter**: one dot per row at (x, y). Shows direction, strength and shape of a relationship, and the outliers that break it.
- **Heatmap**: a grid coloured by value. For correlations: +1 bright, −1 dark, 0 neutral.

## 4. Formula

Only the histogram has one. With $k$ bins over the range $[\min, \max]$:

$$\text{bin width} = \frac{\max - \min}{k}, \qquad \text{count}_j = \#\{x_i : \text{bin}_j \le x_i < \text{bin}_{j+1}\}$$

The choice of $k$ changes the picture. Too few bins hide shape; too many show noise. Try 20, 50 and 100 and look.

## 5. By hand

Eight order values in pounds: 12, 15, 18, 20, 22, 25, 30, 95.

**Histogram, 4 bins.** Range 12–95, width (95−12)/4 = 20.75. Bins: [12, 32.75) → 7 values, [32.75, 53.5) → 0, [53.5, 74.25) → 0, [74.25, 95] → 1. Draw it: a tall bar, two empty bars, a tiny bar far right. *Right-skewed with one outlier.* That one picture tells you the mean will be misleading and the £95 order needs checking.

**Box plot.** Sorted: 12, 15, 18, 20, 22, 25, 30, 95. Median = (20+22)/2 = 21. Q1 = median of lower half (12, 15, 18, 20) = 16.5. Q3 = median of upper half (22, 25, 30, 95) = 27.5. IQR = 11. Upper fence = 27.5 + 1.5 × 11 = 44. The £95 order is above the fence → drawn as a dot. The box runs 16.5–27.5 with a line at 21; whiskers reach 12 and 30.

Sketch both on paper. Then check them in the notebook.

## 6. In code

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4))         # one figure, one axes
monthly.plot(ax=ax)                             # line
ax.set(title="Revenue by month", xlabel="", ylabel="£")

sns.barplot(x=by_country.values, y=by_country.index, ax=ax)      # bar
sns.histplot(order_values, bins=50, ax=ax)                      # histogram
sns.boxplot(data=sales, x="Country", y="Revenue", ax=ax)         # box
sns.scatterplot(data=customers, x="orders", y="spend", ax=ax)    # scatter
sns.heatmap(customers.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1)  # heatmap
```

Three habits that make charts readable: a title that states the finding ("November is the peak"), labelled axes with units, and a log scale (`ax.set_xscale("log")`) when values span several orders of magnitude.

## 7. On real data

The notebook draws each of the six charts on ShopWave data and, for each, writes down the one thing the chart reveals that the table did not: the November spike, the UK's dominance, the extreme right skew of order values, the wholesale outliers per country, the near-straight relationship between orders and spend, and which customer-table columns are redundant.

## 8. Where it breaks

- **Pie charts.** Humans compare lengths well and angles badly. Use a bar.
- **Mean bars without spread.** A bar of average order value hides that half the orders are tiny and three are enormous. Add a box plot.
- **Linear scale on skewed data.** 99% of points squashed into the first pixel. Use log.
- **Too many lines.** Five countries on one line chart is readable; thirty is spaghetti. Pick the top five, group the rest.
- **Truncated axes.** A bar chart starting at £900k makes a 2% dip look like a collapse. Start bars at zero.
- **Correlation heatmap on raw transactions.** Correlation is between *columns of numbers*; compute it on the customer table, not on 400,000 rows of quantity and price.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Line chart of monthly revenue for the top 5 countries *excluding* the UK on one axes, with a legend. Which country is growing?
2. Histogram of `Quantity` per line with 50 bins, then again with a log x-axis. Why is the second one more useful?
3. Box plot of per-customer `spend` for the top 6 countries. Which country has the widest spread? Are the outliers errors?
4. Scatter of `orders` vs `products` on the customer table. Add `alpha=0.3`. What shape do you see and what does it mean?

**Quiz**

1. Which chart for "how are order values distributed"? For "revenue by country"?
2. What do the box, the line inside it, the whiskers and the dots in a box plot represent?
3. Why would you put a log scale on an axis?
4. Give two reasons not to use a pie chart.
5. You see a strong straight-line pattern in a scatter of `orders` vs `spend`. Does that mean one causes the other?

## 10. Recap

- Six charts, six questions: line (time), bar (categories), histogram (distribution), box (spread + outliers), scatter (two variables), heatmap (many variables).
- `fig, ax = plt.subplots()` then draw on `ax`; Seaborn for the chart, Matplotlib for title, labels, scale.
- Title = the finding. Axes = labelled, with units. Skewed data = log scale.
- Every chart in this lesson is a step in the EDA checklist (01.9) and a prerequisite for modelling.

**Next:** [01.8 Statistics you need](../08-statistics-by-hand/): the numbers behind the box plot and the heatmap, by hand.
