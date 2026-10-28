# 01.3 · Pandas: Series, DataFrames, reading files, inspecting data

⏱ about 2 hours · Prerequisites: 01.1, 01.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Operations sends you the file: `online_retail_II.xlsx`, two sheets, half a million rows each. Before anyone asks a question you need to answer the ones you should ask yourself:

- How many rows and columns? What is each column called, and what type is it?
- Which columns have gaps? How many?
- What do the numbers look like: typical quantity, typical price, anything absurd?
- How many distinct customers, products, countries? Which country dominates?

In NumPy, this data would be eight separate arrays with no names, and the text columns (`Description`, `Country`) would not fit at all. You need a table that keeps columns together, names them, allows mixed types across columns, and answers "what is in here?" in a few commands. That is a Pandas **DataFrame**, and the first thing you do with any new dataset, in any job, is the five-minute inspection in this lesson.

For machine learning, this step decides everything that follows: the column types tell you what needs encoding, the gaps tell you what needs imputing, and the absurd values tell you what will wreck your model if you do not fix it first.

## 2. The idea

A **Series** is a NumPy array with a name and a label for every element. A **DataFrame** is a dictionary of Series that share the same labels: one Series per column, all aligned on the row labels (the **index**).

```
                   columns  →   Invoice   Quantity   Price    Country
   index ↓                     ┌─────────┬──────────┬────────┬────────────────┐
     0                         │ 536365  │    6     │  2.55  │ United Kingdom │
     1                         │ 536365  │    6     │  3.39  │ United Kingdom │
     2                         │ 536365  │    8     │  2.75  │ United Kingdom │
                               └─────────┴──────────┴────────┴────────────────┘
                                 int64      int64     float64    object(text)
```

Each column has its own dtype, so a table can hold numbers and text together. Underneath, each numeric column *is* a NumPy array, which is why everything from 01.1 and 01.2 still works: `df["Quantity"] * df["Price"]` is vectorised, `df["Quantity"] > 100` is a mask.

## 3. Definition

- **Series**: a 1-D labelled array. Has `.values` (the NumPy array), `.index` (the labels), `.name`, `.dtype`.
- **DataFrame**: a 2-D table of Series sharing one index. Has `.columns`, `.index`, `.shape`, `.dtypes`.
- **Index**: the row labels. By default 0, 1, 2, …; can be anything (dates, customer IDs).
- **`object` dtype**: Pandas' label for text or mixed columns. A column of strings is `object` (or `string` in newer versions).
- **Missing value**: shown as `NaN` (numbers) or `NaT` (dates); `None` in object columns. All are "missing" to `.isna()`.

The inspection toolkit, in the order you run it:

| Command | Answers |
|---|---|
| `df.shape` | how many rows and columns |
| `df.head()`, `df.tail()`, `df.sample(5)` | what the rows look like, at the start, end, and at random |
| `df.info()` | every column's name, non-null count, dtype, plus memory |
| `df.describe()` | count, mean, std, min, quartiles, max of each numeric column |
| `df.isna().sum()` | missing values per column |
| `df["col"].nunique()`, `df["col"].unique()` | how many distinct values, and which |
| `df["col"].value_counts()` | how often each value appears, most common first |

## 4. Formula

`describe()` reports, for each numeric column with $n$ non-missing values:

$$\text{mean} = \frac{1}{n}\sum x_i \qquad \text{std} = \sqrt{\frac{1}{n-1}\sum (x_i - \text{mean})^2}$$

plus `min`, `25%`, `50%` (median), `75%`, `max`. The quartiles are the values below which 25%, 50%, 75% of the data sit. Lesson 01.8 does these by hand in full; here you only need to read them: **if `max` is a thousand times `75%`, there is an outlier; if `min` is negative where it should not be, there are bad rows.**

`info()` reports, per column, the **non-null count**; missing = `len(df)` − non-null.

## 5. By hand

Five rows from the first ShopWave invoice, with one quantity deliberately missing:

| Invoice | StockCode | Quantity | Price | Customer ID | Country |
|---|---|---|---|---|---|
| 536365 | 85123A | 6 | 2.55 | 17850 | United Kingdom |
| 536365 | 71053 | 6 | 3.39 | 17850 | United Kingdom |
| 536365 | 84406B | *missing* | 2.75 | 17850 | United Kingdom |
| 536366 | 22633 | 6 | 1.85 | *missing* | United Kingdom |
| 536367 | 84879 | 32 | 1.69 | 13047 | France |

**`shape`** → `(5, 6)`.

**`info()`** by hand:

| column | non-null | dtype |
|---|---|---|
| Invoice | 5 | int64 |
| StockCode | 5 | object (text; "85123A" is not a number) |
| Quantity | 4 | float64 (not int, because a missing value forces float) |
| Price | 5 | float64 |
| Customer ID | 4 | float64 (same reason) |
| Country | 5 | object |

**`describe()` for Quantity** (4 values: 6, 6, 6, 32):

- count 4; mean = 50 / 4 = 12.5; min 6; max 32.
- Sorted: 6, 6, 6, 32. Median (50%) = (6 + 6)/2 = 6. 25% = 6, 75% = (6+32)/2 = 19 by the simplest rule (Pandas interpolates slightly differently; the notebook shows its exact answer, 12.5).
- std: deviations from 12.5 are −6.5, −6.5, −6.5, 19.5; squares 42.25 × 3 + 380.25 = 507; divide by $n-1 = 3$ → 169; square root → **13**.

Already the summary tells a story: three ordinary lines and one bulk order pulling the mean to double the median.

**`isna().sum()`** → Quantity 1, Customer ID 1, everything else 0.

**`Country.value_counts()`** → United Kingdom 4, France 1. **`Invoice.nunique()`** → 3.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Build the five-row table from a dictionary and run every inspection command, matching the by-hand answers.
2. See that a column is a Series, that `.values` is the NumPy array underneath, and that masks and arithmetic work as before.
3. Save it to CSV and read it back; read the ShopWave CSVs in `data/`.
4. Read the Excel file and run the same inspection on 541,910 rows.

```python
import pandas as pd
df = pd.read_csv("shopwave_customers_small.csv")
df.shape, df.dtypes
df.info()
df.describe()
df.isna().sum()
df["country"].value_counts()
```

## 7. On real data

The notebook runs the full checklist on the 2010–2011 sheet. You will find: about a quarter of rows have no `Customer ID` (guest checkouts), quantities run from −80,995 to +80,995 (an order and its cancellation), a few prices are zero or absurdly large, and the UK accounts for over 90% of rows. Write these down. They are the to-do list for lesson 01.5 and the reason models trained on raw data fail.

## 8. Where it breaks

- **`info()` prints; it does not return.** `x = df.info()` gives `None`. Use `df.dtypes` and `df.isna().sum()` when you need values.
- **Missing values change dtypes.** An integer column with one `NaN` becomes float. `Customer ID` shows as `17850.0`; it is not a decimal, it is an integer with gaps. Lesson 01.5 fixes this.
- **`describe()` ignores text by default.** `df.describe(include=["object", "str"])` summarises text columns (count, unique, top, freq).
- **Big Excel files are slow.** `read_excel` on 540k rows takes about a minute; CSV takes a second. The notebook saves a CSV copy once so later lessons load fast.
- **`head()` lies by omission.** The first five rows are often the cleanest. Use `sample(5, random_state=0)` as well.
- **Column names with spaces** (`Customer ID`) need `df["Customer ID"]`, never `df.Customer ID`. Prefer bracket access always.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand, then check: for the five-row table, what does `describe()` give for `Price` (count, mean, min, max, median)?
2. Load `shopwave_customers_small.csv`. Which column has the highest standard deviation relative to its mean? What does that tell you?
3. On the real data: how many rows have a missing `Customer ID`? How many distinct `StockCode`s are there, and how many distinct `Description`s? Why might those two numbers differ?
4. `describe(include="object")` on the real data: what is the most frequent `Description`, and how many times does it appear?

**Quiz**

1. What is the relationship between a Series and a NumPy array?
2. Why does a column of whole numbers show as `float64` after loading?
3. Name the command that tells you missing values per column.
4. `describe()` shows `mean` 12.5 and `50%` 6 for the same column. What does that suggest?
5. What does `object` dtype usually mean?

## 10. Recap

- A DataFrame is a table of named, typed columns sharing an index; each column is a Series wrapping a NumPy array.
- `read_csv` / `read_excel` load; `to_csv` saves.
- The inspection checklist: `shape`, `head`/`sample`, `info`, `describe`, `isna().sum()`, `nunique`, `value_counts`. Run it on every new dataset, every time, before anything else.
- Read `describe()` for stories: mean far from median means skew or outliers; impossible mins or maxes mean bad rows.
- What you find here becomes the cleaning plan for 01.5 and the feature plan for Module 02.

**Next:** [01.4 Pandas: selecting, filtering, sorting, new columns](../04-pandas-select-filter/): now that you know what is in the table, pull out exactly the rows and columns a question needs.
