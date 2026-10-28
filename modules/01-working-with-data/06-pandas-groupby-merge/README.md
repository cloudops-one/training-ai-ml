# 01.6 · Pandas: groupby, merge, pivot

⏱ about 2 hours · Prerequisites: 01.5 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The data is clean. Now the questions start:

- *"Revenue per country?"*
- *"Which customers ordered more than five times?"*
- *"Monthly revenue, one column per country, so I can paste it into a slide?"*
- *"Our CRM has each customer's sign-up channel in a separate file. Which channel brings the best customers?"*

Every one of these is "take 400,000 rows and collapse them into a small table of answers", sometimes after joining two tables. Three Pandas operations do all of it: **groupby**, **merge**, **pivot**.

**Why this matters for ML:** models do not learn from raw transactions; they learn from one row per *thing you predict about*. "One row per customer with their total spend, order count and days since last order" is a `groupby`. Adding their sign-up channel from another file is a `merge`. Feature engineering (Module 04) is mostly this lesson applied with intent.

## 2. The idea

**Groupby** is sorting a pile of receipts into folders (one per country), then writing one number on the front of each folder (the total). The three steps have names: **split** into groups, **apply** a function to each, **combine** the results into a table.

**Merge** is looking up each customer in the CRM file and copying their channel onto their orders. A spreadsheet `VLOOKUP`, done for every row at once.

**Pivot** is turning a long list ("UK, January, £500 / UK, February, £600 / DE, January, £200 …") into a grid with months down the side and countries across the top. It is `groupby` with two keys, reshaped for humans.

## 3. Definition

- `df.groupby(keys)[column].agg(func)`: split rows by `keys`, apply `func` to `column` within each group, return one row per group. `func` can be `sum`, `mean`, `count`, `nunique`, `min`, `max`, `first`, or a list of them.
- `pd.merge(left, right, on=key, how=...)`: combine two tables by matching values of `key`.
  - `how="inner"`: keep only keys present in both.
  - `how="left"`: keep every row of `left`; fill missing matches with `NaN`.
  - `how="outer"`: keep everything from both.
- `df.pivot_table(index=rows, columns=cols, values=v, aggfunc=f)`: a two-key groupby displayed as a grid.

## 4. Formula

Groupby-sum written out, for a group $g$ with rows $i \in g$:

$$\text{total}_g = \sum_{i \in g} \text{revenue}_i \qquad\qquad \text{mean}_g = \frac{1}{|g|}\sum_{i \in g} \text{revenue}_i$$

Nothing more. The skill is choosing the right *keys* and the right *function* for the question.

## 5. By hand

Seven clean order lines:

| invoice | customer | country | month | revenue |
|---|---|---|---|---|
| 1001 | C1 | UK | Jan | 50 |
| 1001 | C1 | UK | Jan | 30 |
| 1002 | C2 | DE | Jan | 80 |
| 1003 | C1 | UK | Feb | 40 |
| 1004 | C3 | DE | Feb | 60 |
| 1005 | C2 | DE | Feb | 20 |
| 1006 | C4 | FR | Feb | 90 |

**Groupby country, sum revenue.** Sort into folders: UK = {50, 30, 40} → 120. DE = {80, 60, 20} → 160. FR = {90} → 90.

**Groupby customer: orders and spend.** C1: invoices {1001, 1003} = 2 orders, spend 120. C2: {1002, 1005} = 2, 100. C3: 1, 60. C4: 1, 90. Note `nunique` on invoice, not `count` on rows: C1 has 3 rows but 2 orders.

**Pivot month × country.**

| | UK | DE | FR |
|---|---|---|---|
| Jan | 80 | 80 | 0 |
| Feb | 40 | 80 | 90 |

**Merge with a CRM table.** CRM: C1 → "ads", C2 → "organic", C3 → "ads", C5 → "referral". Left-merge onto the customer summary:

| customer | orders | spend | channel |
|---|---|---|---|
| C1 | 2 | 120 | ads |
| C2 | 2 | 100 | organic |
| C3 | 1 | 60 | ads |
| C4 | 1 | 90 | NaN |

C4 is not in the CRM → `NaN`. C5 is in the CRM but has no orders → not in a *left* merge; it would appear in an *outer* merge. Then groupby channel: ads = (120+60)/2 = £90 average; organic = £100.

## 6. In code

```python
sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False)

customers = sales.groupby("Customer ID").agg(
    orders=("Invoice", "nunique"),
    spend=("Revenue", "sum"),
    last_order=("InvoiceDate", "max"),
)

pd.merge(customers, crm, on="Customer ID", how="left")

sales.pivot_table(index="month", columns="Country", values="Revenue", aggfunc="sum")
```

The named-aggregation form `agg(new_name=("column", "func"))` is the one to learn: it reads like the question.

## 7. On real data

The notebook answers the four questions from section 1 on the 2010–2011 data: the UK is 82% of revenue; 2,845 of 4,338 customers ordered more than once; November 2011 was the peak month at £1.16M; and a merge with a (generated) sign-up-channel table shows how to add external attributes. You also build the **customer table** (one row per customer: orders, spend, first and last order date) that Modules 02–06 reuse as the basis of every feature.

## 8. Where it breaks

- **`count` vs `nunique` vs `size`.** `count` counts non-missing cells, `size` counts rows, `nunique` counts distinct values. "Number of orders" is `nunique("Invoice")`, never `count`.
- **Merge multiplies rows.** If the key is not unique on the right side, each left row matches several right rows and your table silently grows. Always check `len()` before and after, or use `validate="one_to_one"` / `"many_to_one"`.
- **Key dtype mismatch.** `Customer ID` as `17850.0` (float) will not match `17850` (int). Cast first.
- **Groupby drops NaN keys by default.** Rows with a missing country vanish from the totals. Use `dropna=False` if they matter.
- **Pivot with duplicate index/column pairs** needs `pivot_table` (which aggregates), not `pivot` (which errors).
- **Mean of means.** The average of per-country averages is not the overall average. Aggregate from rows, not from aggregates.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Average order value (revenue per invoice) per country, for countries with at least 50 invoices. Which is highest?
2. For each customer, the number of distinct products bought. Plot the distribution (histogram). What does it tell you about ShopWave's buyers?
3. Pivot: revenue by weekday (rows) × hour of day (columns). When should the support desk be fully staffed?
4. Build a tiny `products` table (StockCode → most common Description) and merge it onto a per-product revenue summary. Validate the merge did not change the row count.

**Quiz**

1. The three steps of groupby, in order?
2. "How many orders did each customer place": `count`, `size` or `nunique`, and on which column?
3. What happens to a left row with no match in a left merge? In an inner merge?
4. Your merge returned more rows than the left table. What went wrong?
5. When do you need `pivot_table` instead of `pivot`?

## 10. Recap

- **Groupby** = split → apply → combine. One row per group. Use named aggregation.
- **Merge** = join on a key. Choose `how` deliberately; check row counts.
- **Pivot** = two-key groupby shown as a grid.
- The customer table (`groupby("Customer ID")`) is the backbone of every ML feature in this course.
- `nunique` for orders, cast keys to the same dtype, beware silent row multiplication.

**Next:** [01.7 Plotting](../07-plotting/): the six charts that turn these tables into understanding.
