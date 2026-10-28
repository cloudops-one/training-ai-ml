# Pandas

Examples use ShopWave's transaction columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`. Learn it in [01.3](../modules/01-working-with-data/03-pandas-basics/) → [01.6](../modules/01-working-with-data/06-pandas-groupby-merge/).

```python
import numpy as np
import pandas as pd
```

## Load and save

| Task | Code |
|---|---|
| Read CSV / Excel / Parquet | `pd.read_csv(path)` · `pd.read_excel(path, sheet_name="Year 2010-2011")` · `pd.read_parquet(path)` |
| Course data (fast) | `from download import load_retail; raw = load_retail()` → [data/README](../data/README.md) |
| Parse dates on load | `pd.read_csv(path, parse_dates=["InvoiceDate"])` |
| Write | `df.to_csv("out.csv", index=False)` · `df.to_parquet("out.parquet")` |

## Inspect

| Task | Code |
|---|---|
| First / last rows | `df.head()` · `df.tail(3)` · `df.sample(5, random_state=0)` |
| Size, types, missing | `df.shape` · `df.dtypes` · `df.info()` |
| Summary stats | `df.describe()` · `df.describe(include="str")` (text columns) |
| Count values | `df["Country"].value_counts()` · `df["Country"].value_counts(normalize=True)` |
| Distinct values | `df["Customer ID"].nunique()` · `df["Country"].unique()` |
| Missing per column | `df.isna().sum()` · `df.isna().mean()` (share) |

## Select

| Task | Code |
|---|---|
| One column (Series) / several (DataFrame) | `df["Price"]` · `df[["Quantity", "Price"]]` |
| Rows by label / position | `df.loc[10:20, "Price"]` · `df.iloc[0:5, 0:3]` |
| One cell | `df.at[0, "Price"]` |
| Columns by type | `df.select_dtypes("number")` |

## Filter

```python
uk = df[df["Country"] == "United Kingdom"]
big = df[(df["Quantity"] > 0) & (df["Price"] > 10)]              # & and |, with brackets
some = df[df["Country"].isin(["France", "Germany"])]
cancelled = df[df["Invoice"].astype(str).str.startswith("C")]
mugs = df[df["Description"].str.contains("MUG", na=False)]
q = df.query("Quantity > 0 and Price > 0")                        # same idea, as text
```

## Create and change columns

```python
df["Revenue"] = df["Quantity"] * df["Price"]                      # vectorised, no loop
df["size"] = pd.cut(df["Revenue"], bins=[0, 10, 100, float("inf")], labels=["S", "M", "L"])
df["is_uk"] = (df["Country"] == "United Kingdom").astype(int)
df = df.rename(columns={"Customer ID": "customer_id"})
df = df.drop(columns=["StockCode"])
df = df.assign(log_rev=lambda d: d["Revenue"].clip(lower=0).add(1).pipe(np.log))
```

## Clean

| Task | Code |
|---|---|
| Drop rows with missing values | `df.dropna(subset=["Customer ID"])` |
| Fill missing | `df["Description"] = df["Description"].fillna("UNKNOWN")` |
| Duplicates | `df.duplicated().sum()` · `df = df.drop_duplicates()` |
| Change type | `df["Customer ID"] = df["Customer ID"].astype(int)` |
| Text tidy | `df["Description"].str.strip().str.upper()` |
| Clip outliers | `df["Revenue"].clip(upper=df["Revenue"].quantile(0.99))` |
| Sort / rank | `df.sort_values("Revenue", ascending=False)` · `df["Revenue"].rank(pct=True)` |

The course's standard cleaning (cancellations, returns, guest checkouts, duplicates) is `clean_retail()` in [01.5](../modules/01-working-with-data/05-pandas-clean/).

## Dates

```python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")
df["month"] = df["InvoiceDate"].dt.to_period("M")
df["weekday"] = df["InvoiceDate"].dt.day_name()
df["hour"] = df["InvoiceDate"].dt.hour
days_since = (pd.Timestamp("2011-12-10") - df["InvoiceDate"]).dt.days
weekly = df.set_index("InvoiceDate")["Revenue"].resample("W").sum()   # "D", "W", "ME"
rolling = weekly.rolling(4).mean()
```

## Group and aggregate

```python
df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)

customers = df.groupby("Customer ID").agg(
    orders=("Invoice", "nunique"),
    spend=("Revenue", "sum"),
    last_order=("InvoiceDate", "max"),
)
df["share_of_invoice"] = df["Revenue"] / df.groupby("Invoice")["Revenue"].transform("sum")
```

## Combine and reshape

| Task | Code |
|---|---|
| Join tables on a key | `customers.merge(segments, on="Customer ID", how="left")` |
| Check a join | `merge(..., validate="one_to_one", indicator=True)` |
| Stack rows | `pd.concat([df_2010, df_2011], ignore_index=True)` |
| Pivot (wide) | `df.pivot_table(index="Country", columns="month", values="Revenue", aggfunc="sum", fill_value=0)` |
| Crosstab | `pd.crosstab(df["Country"], df["weekday"], normalize="index")` |
| Wide → long | `wide.melt(id_vars="Country", var_name="month", value_name="Revenue")` |

## Pandas 3 notes

- **No chained assignment.** `df["Price"].fillna(0, inplace=True)` does nothing in pandas 3. Write `df["Price"] = df["Price"].fillna(0)`.
- **Copy-on-write.** A filtered DataFrame behaves like a copy; changing it never changes the original. No more `SettingWithCopyWarning`.
- **Text columns have dtype `str`** (not `object`). Use `describe(include="str")`; missing text is `NaN`.
- **Mixed-type columns from Excel** (numbers and text in `Invoice`): convert with `.astype(str)` before `.str` methods.
- **Date parts:** `.dt.day_of_week`, `.dt.day_of_year` (the old `dayofweek` spellings are deprecated). Month-end resample alias is `"ME"`.

## Speed

- Vectorise: column arithmetic is about **100×** faster than a Python loop over rows ([01.1](../modules/01-working-with-data/01-numpy-arrays/)).
- Avoid `df.apply(..., axis=1)` on big tables when a column expression works.
- Save cleaned data to Parquet: it keeps types and loads in about a second.
