# 01.4 · Pandas: selecting, filtering, sorting, new columns

⏱ about 2 hours · Prerequisites: 01.3 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Now that you know what is in the table, the questions arrive:

- Marketing: *"List every German order line over £50, biggest first, with just the invoice, product and value."*
- Finance: *"Add a revenue column (quantity × price) so we stop recalculating it in Excel."*
- Operations: *"Show me the cancelled invoices from last week."* (Invoices starting with `C`.)

Each is the same four moves: **pick columns**, **pick rows by condition**, **sort**, **add a derived column**. In machine learning you do exactly these moves to build a training set: choose the feature columns, filter to the rows you trust, derive new features, sort by date before splitting. Get fluent here and Module 02 is about ideas, not syntax.

## 2. The idea

A DataFrame answers two different questions, and Pandas gives them different syntax so you never confuse them:

- **"Which columns?"** → `df["col"]` or `df[["col1", "col2"]]`.
- **"Which rows?"** → a boolean mask, `df[df["Quantity"] > 100]`, exactly the NumPy mask from 01.2 but with a column name in it.

Put them together with `.loc[rows, columns]`: *"rows where this is true, and these columns"*. `.loc` uses **labels** (column names, index labels); its sibling `.iloc` uses **positions** (0, 1, 2, …), like NumPy.

```
df.loc[ df["Country"] == "Germany" ,  ["Invoice", "Description", "revenue"] ]
          ↑ which rows (a mask)            ↑ which columns (a list of names)
```

Sorting reorders rows by one or more columns. A new column is just assignment: `df["revenue"] = df["Quantity"] * df["Price"]`, vectorised across every row at once.

## 3. Definition

| Move | Syntax | Returns |
|---|---|---|
| one column | `df["col"]` | Series |
| several columns | `df[["a", "b"]]` (note the double brackets: a list inside) | DataFrame |
| rows by condition | `df[mask]` where `mask = df["col"] > 5` | DataFrame |
| rows and columns by label | `df.loc[mask, ["a", "b"]]` | DataFrame |
| rows and columns by position | `df.iloc[0:5, 0:3]` | DataFrame |
| membership | `df["col"].isin(["UK", "DE"])` | mask |
| range | `df["col"].between(10, 20)` | mask (inclusive both ends) |
| text match | `df["col"].str.startswith("C")`, `.str.contains("HEART")` | mask |
| sort | `df.sort_values("col", ascending=False)`; several: `sort_values(["a", "b"])` | DataFrame |
| new column | `df["new"] = expression` | modifies `df` |
| new column, chainable | `df.assign(new=expression)` | new DataFrame |
| top / bottom n | `df.nlargest(10, "col")`, `df.nsmallest(10, "col")` | DataFrame |

Masks combine with `&`, `|`, `~` and brackets, as in NumPy.

## 4. Formula

Nothing new mathematically; the derived column is vectorised arithmetic:

$$\text{revenue}_i = \text{Quantity}_i \times \text{Price}_i \quad \text{for every row } i$$

What matters is the **order of operations** for a typical request: filter first (fewer rows), then derive (less work), then sort (on the smaller set), then select columns (what the reader wants). Pandas lets you write it as a chain:

```python
(df[df["Country"] == "Germany"]
   .assign(revenue=lambda d: d["Quantity"] * d["Price"])
   .query("revenue > 50")
   .sort_values("revenue", ascending=False)
   [["Invoice", "Description", "revenue"]])
```

## 5. By hand

Six order lines:

| # | Invoice | Description | Quantity | Price | Country |
|---|---|---|---|---|---|
| 0 | 536365 | WHITE HANGING HEART | 6 | 2.55 | United Kingdom |
| 1 | 536367 | ASSORTED COLOUR BIRD | 32 | 1.69 | France |
| 2 | 536370 | RED TOADSTOOL LED | 24 | 1.65 | Germany |
| 3 | 536370 | ALARM CLOCK BAKELIKE | 48 | 3.75 | Germany |
| 4 | C536379 | DISCOUNT | −1 | 27.50 | Germany |
| 5 | 536381 | PINK HEART SHAPE | 12 | 1.25 | United Kingdom |

**Finance: add revenue.** Multiply row by row: 15.30, 54.08, 39.60, 180.00, −27.50, 15.00.

**Marketing: German lines over £50, biggest first, three columns.**

1. Mask `Country == "Germany"`: rows 2, 3, 4.
2. Mask `revenue > 50`: of those, only row 3 (180.00). Row 2 is 39.60, row 4 is negative.
3. Sort: one row, nothing to sort.
4. Columns: `536370 | ALARM CLOCK BAKELIKE | 180.00`.

**Operations: cancelled invoices.** `Invoice.str.startswith("C")` → row 4 only.

**Sort everything by revenue descending**: rows 3, 1, 2, 0, 5, 4 (180.00, 54.08, 39.60, 15.30, 15.00, −27.50).

**`iloc[1:3]`** → rows at positions 1 and 2 (France, first Germany). **`loc[1:3]`** → rows with *labels* 1, 2 **and 3** (`.loc` slices are inclusive; `.iloc` slices are not). That difference catches everyone once.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Build the six-row table and reproduce every by-hand answer.
2. See the difference between `df["col"]` (Series) and `df[["col"]]` (DataFrame), and between `.loc` and `.iloc`.
3. Use `isin`, `between`, `.str` methods, `nlargest`.
4. Write the marketing request as one readable chain.
5. Learn the one warning everyone hits (`SettingWithCopyWarning`) and how to avoid it.

```python
germany = df[df["Country"] == "Germany"]
df["revenue"] = df["Quantity"] * df["Price"]
df.loc[df["revenue"] > 50, ["Invoice", "Description", "revenue"]].sort_values("revenue", ascending=False)
```

## 7. On real data

The notebook answers all three requests on the 2010–2011 sheet: Germany's biggest lines, a permanent `revenue` column, and the cancelled invoices (there are over 9,000 of them; you will count and value them). Then marketing's follow-up: *"Which products containing the word HEART sold most units in the UK?"*, which is `str.contains` + mask + sort in three lines.

## 8. Where it breaks

- **Single vs double brackets.** `df["a"]` is a Series; `df[["a"]]` is a one-column DataFrame. Models want a DataFrame for `X` and a Series for `y`; you will meet this in Module 02.
- **`SettingWithCopyWarning`.** `sub = df[df["x"] > 1]; sub["y"] = 0` may or may not modify `df`. Pandas warns. Fix: `sub = df[df["x"] > 1].copy()` before assigning, or use `df.loc[mask, "y"] = 0` on the original.
- **`.loc` slices include the end label; `.iloc` slices exclude the end position.**
- **`and`/`or` on masks** raise an error, as in NumPy. Use `&`/`|` with brackets.
- **`.str` on a numeric column** fails. `Invoice` is read as text here because some values start with `C`; if a column is numeric you must `.astype(str)` first.
- **Chained indexing** `df["a"]["b"]` works for reading but not reliably for writing. Always `df.loc[row, col]` to write.
- **`sort_values` returns a new DataFrame**; it does not sort in place unless you ask. Assign the result.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand, then code: from the six-row table, select UK rows with quantity ≥ 10, showing Description and Quantity.
2. Add a column `is_cancelled` (True when Invoice starts with `C`). How many cancelled lines, and what is their total revenue?
3. Real data: the five most expensive single items (highest `Price`) that are *not* cancellations and have quantity exactly 1. What are they?
4. Real data: build a table of only France, Germany and Netherlands lines with revenue between £20 and £100, sorted by country then revenue descending. How many rows?

**Quiz**

1. What is the difference between `df["Price"]` and `df[["Price"]]`?
2. Write a mask for "Country is UK or France".
3. `df.loc[2:4]` returns how many rows? And `df.iloc[2:4]`?
4. You filtered a DataFrame, assigned a new column to the result, and saw `SettingWithCopyWarning`. What is the fix?
5. Why does Pandas read the `Invoice` column as text rather than numbers?

## 10. Recap

- Columns: `df["a"]` (Series), `df[["a", "b"]]` (DataFrame). Rows: a boolean mask. Both: `df.loc[mask, cols]`.
- `.loc` is by label (end-inclusive); `.iloc` is by position (end-exclusive).
- `isin`, `between`, `.str.contains`/`startswith` build masks for the common cases.
- `sort_values` reorders; `nlargest` is the shortcut for "top n". Both return new frames.
- `df["new"] = ...` derives a column across all rows at once. Copy a filtered frame before writing to it.

**Next:** [01.5 Pandas: missing values, duplicates, data types, dates](../05-pandas-clean/): the rows you found in 01.3 that are wrong, and how to fix them.
