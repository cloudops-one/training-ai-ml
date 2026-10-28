# 01.P · Mini-project: the ShopWave sales report

⏱ about 2 hours · Prerequisites: all of Module 01 · Notebook: [project.ipynb](project.ipynb) · Solution: [solution.ipynb](solution.ipynb)

---

## The brief

It is the end of your first month. The CEO wants one document: *"Tell me what happened in the last year, with numbers I can trust and five things I should do about it."*

You have the 2010–2011 transactions. Produce a notebook that a non-technical reader could follow from top to bottom: a short sentence of intent before each step, a chart or table, and a sentence saying what it shows.

## Tasks

1. **Clean the data** and justify each decision in a markdown cell: cancellations, negative quantities, zero prices, missing customer IDs, duplicates. Report rows before and after, and the revenue you are *excluding* by dropping guest checkouts (the CEO will ask).
2. **Revenue by month**: a line chart. State the growth from the first full month to the last full month, and name the peak. Explain why December 2011 looks low.
3. **Top 10 products by revenue**: a horizontal bar chart. Note any rows that are not real products (postage, manual adjustments) and decide whether to exclude them.
4. **Top 10 countries by revenue excluding the UK**: a bar chart, plus the UK's share as one number. Add average order value per country for the top 5; comment on the wholesale pattern.
5. **Repeat vs one-time customers**: how many of each, and what share of revenue each group brings. One chart.
6. **Weekday and hour pattern**: a weekday × hour pivot shown as a heatmap. One sentence on staffing.
7. **Five business insights**: a final markdown cell with five numbered, specific, numeric recommendations. "Sales are good in November" is not an insight. "November is 15% of annual revenue; start the Christmas stock build-up in August, when the ramp begins" is.

## Rubric (self-assess, 20 points)

| Criterion | Points |
|---|---|
| Cleaning decisions are stated and justified; before/after counts shown | 4 |
| Every chart has a title that states the finding, labelled axes, and sensible scale (log where needed) | 4 |
| Numbers are correct and reproducible (notebook runs top to bottom with no errors) | 4 |
| Medians used where distributions are skewed; outliers acknowledged, not silently deleted | 3 |
| Five insights are specific, quantified and actionable | 5 |

16+ means you are ready for Module 02. Below 12, redo tasks 1, 2 and 7 after rereading 01.5, 01.7 and 01.8.

## How to work

- Start from [project.ipynb](project.ipynb); every task has a cell with `# TODO`.
- Do not open the solution until you have attempted all seven tasks. Then compare, and note what the solution did that you did not.
- Your numbers may differ slightly from the solution if your cleaning decisions differ. That is fine *if you wrote the decisions down*.

## What you are practising

Everything in Module 01: cleaning (01.5), groupby and pivot (01.6), the six charts (01.7), mean vs median and outliers (01.8), and the discipline of writing findings as sentences (01.9). In Module 02 the same customer data becomes the input to your first model.
