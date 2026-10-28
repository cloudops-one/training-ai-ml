# Module 01 · Working with Data

**ShopWave asks:** *"We have two years of orders in a spreadsheet. Nobody can answer a simple question about them without an afternoon of copy-pasting. Make sense of our sales data."*

**Time:** about 20 hours. **Prerequisites:** Module 00, Python basics (lists, dicts, loops, functions).

## Where you are

Finance wants revenue by month. Marketing wants the top customers in Germany. Operations wants to know how many orders were cancelled last December. The data exists: 540,000 order lines, one per product per invoice. But it lives in Excel, it has missing customer IDs, negative quantities for returns, duplicated rows, and dates stored as text.

Every machine-learning model in this course starts from a clean table of numbers. **Getting from a raw spreadsheet to that table is most of the job**, and it is the skill employers test first. This module gives you the three tools every data person uses daily: NumPy for fast arithmetic on arrays, Pandas for tables, and Matplotlib/Seaborn for pictures. It ends with a repeatable checklist for exploring any new dataset and a real sales report for ShopWave.

## What you will be able to do afterwards

- Explain why NumPy arrays are faster than Python lists and use shapes, indexing, masks and broadcasting without guessing.
- Load CSV and Excel files into a DataFrame, inspect them in one minute, and select, filter, sort and derive columns.
- Find and fix missing values, duplicates, wrong types and bad dates.
- Summarise with `groupby`, combine tables with `merge`, reshape with `pivot`.
- Draw the six charts that answer most business questions, and know which one to use.
- Compute mean, median, standard deviation, percentiles and correlation by hand, and know what each is for.
- Run a repeatable EDA checklist on any new dataset and write down what you found.
- Deliver a sales report for ShopWave from the raw data.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 01.1 | [NumPy: arrays, shapes and why vectors beat loops](01-numpy-arrays/) | 2 h |
| 01.2 | [NumPy: indexing, boolean masks, broadcasting, random numbers](02-numpy-indexing-broadcasting/) | 2 h |
| 01.3 | [Pandas: Series, DataFrames, reading files, inspecting data](03-pandas-basics/) | 2 h |
| 01.4 | [Pandas: selecting, filtering, sorting, new columns](04-pandas-select-filter/) | 2 h |
| 01.5 | [Pandas: missing values, duplicates, data types, dates](05-pandas-clean/) | 2 h |
| 01.6 | [Pandas: groupby, merge, pivot](06-pandas-groupby-merge/) | 2 h |
| 01.7 | [Plotting with Matplotlib and Seaborn: the six charts you need](07-plotting/) | 2 h |
| 01.8 | [Statistics you need: mean, median, spread, percentiles, correlation, by hand](08-statistics-by-hand/) | 2 h |
| 01.9 | [Exploratory data analysis: a repeatable checklist](09-eda-checklist/) | 2 h |
| 01.P | [Mini-project: the ShopWave sales report](project/) | 2 h |

## The maths in this module

Only arithmetic on tables: sums, averages, differences from the average, and one formula (correlation) that combines them. Lesson 01.8 does all of it by hand on ten rows. If you can add a column of numbers and divide, you have what you need.

## Why so much time on data before any "real" machine learning?

Because in Module 02 you will feed a model a table, and if that table has a missing value, a text column, or a date stored as a string, the model will refuse to run or, worse, run and quietly learn nonsense. Practitioners typically report spending most of a project's time on exactly the work in this module. Learning it well now means every later module is about the ideas, not about fighting Pandas.
