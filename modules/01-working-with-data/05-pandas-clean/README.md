# 01.5 · Pandas: missing values, duplicates, data types, dates

⏱ about 2 hours · Prerequisites: 01.3, 01.4 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The finance manager asks for last year's total revenue. You load ShopWave's 541,910 transactions, multiply quantity by price, sum, and send the number. An hour later: *"That's wrong. It includes 9,000 cancelled orders, thousands of returns with negative quantities, and 135,000 guest checkouts we can't attribute to anyone. Also, half the product names are missing and the dates came in as text."*

Every dataset you will ever get is like this. Before any chart, any statistic, any model, you must **clean**: find what is missing, what is duplicated, what is the wrong type, and decide what to do about each. In this course roughly a third of all the work is cleaning. Models trained on dirty data learn dirt.

**Why this matters for ML:** a model cannot take `NaN` as input, treats `"12"` (text) differently from `12` (number), and will happily learn that "cancelled orders have negative revenue" as if that were a customer behaviour. Lesson 04.3 does imputation properly; this lesson teaches you to *see* the problems first.

## 2. The idea

Think of a warehouse stock-take. You do not start counting; you first walk every aisle asking four questions: *What is empty? What is counted twice? What is labelled wrong? What is in the wrong units?* Only then do you count.

Pandas gives you one method for each question:

| Question | Method |
|---|---|
| What is empty? | `isna()` |
| What is counted twice? | `duplicated()` |
| What is labelled wrong (type)? | `dtypes`, `astype()` |
| What is in the wrong units / format (dates)? | `to_datetime()`, `.dt` |

## 3. Definition

- **Missing value**: a cell with no value. Pandas shows it as `NaN` (not a number) or `NaT` (not a time). `None` also counts.
- **Duplicate row**: a row identical to an earlier one across all (or chosen) columns. Usually an export glitch; sometimes a real repeat order, so check before deleting.
- **dtype**: the data type of a column: `int64`, `float64`, `bool`, `datetime64`, `object` (mixed / text), `str`, `category`. The wrong dtype makes arithmetic fail or silently do the wrong thing.
- **Datetime**: a proper timestamp. Once a column is datetime you can ask for its year, month, weekday, hour, or subtract two dates to get a duration.

Three ways to handle a missing value, each a decision not a reflex:

1. **Drop the row** when the missing field is essential (no customer ID → cannot analyse the customer).
2. **Drop the column** when most of it is missing and it is not important.
3. **Fill it** (impute) with a sensible value: 0, the median, "unknown", or a value derived from other columns.

## 4. Formula

No equations in this lesson. One rule to memorise instead:

> **Missing % = missing count ÷ total rows × 100.**  
> Under 5%: dropping rows is usually fine. 5–30%: fill, carefully. Over 30%: question whether the column is usable at all.

## 5. By hand

Six ShopWave orders, as exported:

| row | invoice | product | qty | price | customer | date |
|---|---|---|---|---|---|---|
| 0 | 536365 | WHITE LANTERN | 6 | 3.39 | 17850 | 2010-12-01 |
| 1 | 536365 | WHITE LANTERN | 6 | 3.39 | 17850 | 2010-12-01 |
| 2 | C536379 | MUG | −1 | 2.55 | 17850 | 2010-12-01 |
| 3 | 536380 | *(blank)* | 2 | 0.00 | *(blank)* | 2010-12-01 |
| 4 | 536381 | JAM JAR | 12 | "1.25" | 15311 | "01/12/2010" |
| 5 | 536382 | TEA TOWEL | 3 | 1.65 | 15311 | 2010-12-02 |

Walk the four questions:

- **Empty?** Row 3 has no product and no customer. That is 2 cells missing in 36 → 5.6%. Row 3 is useless for customer analysis: drop it.
- **Duplicated?** Rows 0 and 1 are identical. Same invoice, same product, same quantity. An export glitch; keep one.
- **Wrong type?** Row 4's price is the *text* `"1.25"`. If the column is text, `qty × price` fails. Convert the column to a number.
- **Wrong format?** Row 4's date is `"01/12/2010"`, a different format. Convert the whole column to datetime so every row is comparable.
- **Business rule, not a type problem:** row 2's invoice starts with `C` and quantity is −1. That is a cancellation. Decide: keep for a returns analysis, drop for a sales analysis.

After cleaning for a sales report: rows 0, 4, 5 remain. Revenue = 6×3.39 + 12×1.25 + 3×1.65 = 20.34 + 15.00 + 4.95 = **£40.29**. Done naively on all six rows (treating "1.25" as 0 because text cannot multiply) you would have reported 20.34×2 − 2.55 + 0 + 4.95 = £43.08. Wrong number, wrong direction, and nobody would notice.

## 6. In code

```python
df.isna().sum()                         # missing per column
df.isna().mean().mul(100).round(1)      # missing %
df.duplicated().sum()                   # exact duplicate rows
df = df.drop_duplicates()
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")   # text → number, bad → NaN
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["month"] = df["InvoiceDate"].dt.to_period("M")
df = df.dropna(subset=["Customer ID"])
df["Description"] = df["Description"].fillna("UNKNOWN")
```

Each of these appears in [lesson.ipynb](lesson.ipynb) on the six-row table first, then on the real data.

## 7. On real data

On the full 2010–2011 sheet the notebook finds: 135,080 missing customer IDs (24.9%), 1,454 missing descriptions, 5,268 exact duplicates, 9,288 cancellations, 10,624 negative quantities, 2,517 zero-or-negative prices, and a `Customer ID` stored as a decimal (`17850.0`) because missing values forced the column to float. You will build `clean_retail()`, the function every later lesson reuses, and watch 541,910 rows become 392,693.

## 8. Where it breaks

- **Dropping too eagerly.** Dropping all rows with a missing customer ID throws away a quarter of ShopWave's revenue. Fine for a *customer* analysis; wrong for a *total revenue* figure. Clean for the question you are answering.
- **Missing is not random.** Guest checkouts (missing ID) may be a different kind of customer. If you drop them, your customer model never sees them.
- **`fillna(0)` lies.** A missing price is not a free product. Fill with something defensible or mark it as missing with a separate flag column.
- **Duplicates that are real.** Two identical rows can be a customer buying the same item twice in one order. Check `Invoice` + `StockCode` + `Quantity` before deleting.
- **`errors="coerce"` hides problems.** It turns unparseable values into NaN silently. Count NaNs before and after to see what it ate.
- **Dates without timezone or with mixed formats.** `01/02/2011` is 1 Feb in Europe and 2 Jan in the US. Specify `format=` or `dayfirst=True`.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. On the raw data, what percentage of *revenue* (not rows) comes from guest checkouts? Should the finance report drop them?
2. Hundreds of stock codes have more than one description. Find them and pick a rule for choosing one.
3. Add a `is_cancelled` boolean column instead of dropping cancellations. How much revenue was cancelled in November 2011?
4. Convert `Country` to the `category` dtype. Compare `memory_usage(deep=True)` before and after.

**Quiz**

1. What does `isna().mean()` give you?
2. Why is `Customer ID` a float column in the raw file?
3. Name two reasons not to `drop_duplicates()` blindly.
4. What does `errors="coerce"` do, and what is the risk?
5. Give one case where filling a missing value with 0 is correct and one where it is wrong.

## 10. Recap

- Four questions before any analysis: missing? duplicated? wrong type? wrong format?
- `isna()`, `duplicated()`, `dtypes` / `astype()` / `to_numeric()`, `to_datetime()` answer them.
- Every fix is a *decision*: drop, fill, or flag. Write the decision down.
- Clean for the question you are answering; a revenue total and a customer model need different cleaning.
- `clean_retail()` is now your reusable ShopWave cleaning function.

**Next:** [01.6 Groupby, merge, pivot](../06-pandas-groupby-merge/): turning clean rows into answers.
