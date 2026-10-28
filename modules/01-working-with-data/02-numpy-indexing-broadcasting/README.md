# 01.2 · NumPy: indexing, boolean masks, broadcasting, random numbers

⏱ about 2 hours · Prerequisites: 01.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Three requests land in one morning:

- Marketing: *"Give me only the order lines worth more than £100."* You have a revenue array of 541,910 numbers. You need to pick rows by a condition.
- Pricing: *"Here are our 3 product prices and 4 bulk-discount tiers. Build the full 3 × 4 price table."* Two small arrays of different shapes must combine into a grid.
- Finance: *"Simulate 1,000 possible December revenues so we can plan stock."* You need random numbers, and the same random numbers tomorrow so the report is reproducible.

All three are everyday array work. In machine learning the same three moves are everywhere: masks select training rows, broadcasting is how a model applies its weights to every row at once, and seeded random numbers are how you split data and initialise models so results can be repeated.

## 2. The idea

**Indexing** is asking an array for a piece of itself by position: the first element, the last ten, the cell at row 2 column 3.

**Masking** is asking by condition instead of position: "all elements greater than 100". You write the condition, NumPy returns an array of `True`/`False` the same shape, and you use that as the index. Think of laying a stencil over the data and keeping only what shows through the holes.

**Broadcasting** is NumPy's rule for arithmetic between arrays whose shapes differ. A scalar is "stretched" to match an array; a 1-D row is stretched down a 2-D grid; a column is stretched across. No copy is made; NumPy just reads the smaller array repeatedly.

```
   prices (3,1)         tiers (1,4)              result (3,4)
   [ 2.50 ]       ×   [1.0 0.9 0.8 0.7]   =   [2.50 2.25 2.00 1.75]
   [ 3.40 ]                                    [3.40 3.06 2.72 2.38]
   [ 2.80 ]                                    [2.80 2.52 2.24 1.96]
```

**Random numbers with a seed** are a recipe: same seed, same sequence, every time, on every machine. That is what makes an experiment reproducible.

## 3. Definition

- **Index**: `a[i]` is the element at position `i`, counting from 0; negative counts from the end. For 2-D, `a[r, c]`.
- **Slice**: `a[start:stop:step]`, stop exclusive. `a[:3]` first three, `a[-3:]` last three, `a[::2]` every other. 2-D: `a[:, 0]` whole first column; `a[1, :]` whole second row.
- **View**: a slice does not copy; it looks at the same memory. Changing the slice changes the original. Use `.copy()` when you need independence.
- **Boolean mask**: an array of `True`/`False` with the same shape as the data. `a[mask]` returns the elements where the mask is `True`, as a new 1-D array (a copy).
- **Fancy indexing**: `a[[0, 4, 2]]` picks elements by a list of positions, in that order.
- **Broadcasting**: the rule by which arrays of different shapes are combined element-wise (formula below).
- **Generator**: `rng = np.random.default_rng(seed)`; `rng.integers`, `rng.random`, `rng.normal`, `rng.choice` produce reproducible random numbers.

## 4. Formula

**Broadcasting rule.** Compare the two shapes **from the right**, one dimension at a time. Two dimensions are compatible if they are **equal**, or **one of them is 1**. A missing dimension counts as 1. If every pair is compatible, the result has the larger of each pair; otherwise NumPy raises an error.

```
(3, 4) × (4,)     →  (3, 4) × (1, 4)  → 4=4 ✓, 3 vs 1 ✓ → (3, 4)
(3, 4) × (3,)     →  (3, 4) × (1, 3)  → 4≠3 ✗            → ValueError
(3, 1) × (1, 4)   →  1 vs 4 ✓, 3 vs 1 ✓                  → (3, 4)
(3, 4) × scalar   →  scalar is ()  → treated as (1, 1)   → (3, 4)
```

**Counting with masks.** Because `True` behaves as 1 and `False` as 0:

$$\text{count}(\text{mask}) = \sum \text{mask} \qquad \text{fraction}(\text{mask}) = \text{mean}(\text{mask})$$

## 5. By hand

**Indexing.** `rev = [15.30, 20.34, 22.00, 20.34, 20.34]` (the five line revenues from 01.1).

- `rev[0]` → 15.30. `rev[-1]` → 20.34. `rev[1:3]` → `[20.34, 22.00]` (positions 1 and 2; 3 is excluded). `rev[::2]` → `[15.30, 22.00, 20.34]`.

**Mask.** Condition `rev > 20`:

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rev | 15.30 | 20.34 | 22.00 | 20.34 | 20.34 |
| rev > 20 | F | T | T | T | T |

`rev[rev > 20]` → `[20.34, 22.00, 20.34, 20.34]`. Count = 4. Fraction = 4/5 = 0.8.

**Broadcasting.** Prices of 3 products, discounts for 4 tiers:

prices (shape `(3, 1)`): 2.50, 3.40, 2.80. Tiers (shape `(1, 4)`): ×1.0, ×0.9, ×0.8, ×0.7.

Shapes `(3, 1)` and `(1, 4)`: from the right, 1 vs 4 ✓, 3 vs 1 ✓ → result `(3, 4)`.

| | tier 1 (×1.0) | tier 2 (×0.9) | tier 3 (×0.8) | tier 4 (×0.7) |
|---|---|---|---|---|
| £2.50 | 2.50 | 2.25 | 2.00 | 1.75 |
| £3.40 | 3.40 | 3.06 | 2.72 | 2.38 |
| £2.80 | 2.80 | 2.52 | 2.24 | 1.96 |

Twelve multiplications, no loop, and the code is one line: `prices[:, None] * tiers`.

**Would `(3,) × (4,)` work?** From the right: 3 vs 4, neither is 1 → error. You must first turn prices into a column, `(3, 1)`. That reshape is the whole trick.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Index and slice the five revenues; see that a slice is a view.
2. Build the `rev > 20` mask, select with it, count with it, combine masks with `&`, `|`, `~`.
3. Build the 3 × 4 price table with broadcasting; make `(3,) × (4,)` fail on purpose and read the error.
4. Draw random numbers with a seed and watch them repeat.

```python
import numpy as np
rev = np.array([15.30, 20.34, 22.00, 20.34, 20.34])
rev[rev > 20]                       # mask
prices = np.array([2.50, 3.40, 2.80])
tiers = np.array([1.0, 0.9, 0.8, 0.7])
prices[:, None] * tiers             # (3,1) × (4,) → (3,4)
rng = np.random.default_rng(42)
rng.integers(1, 7, size=5)          # five dice rolls, same every run
```

## 7. On real data

The notebook loads the 2010–2011 sheet and answers marketing's question with a mask (`revenue > 100`), then narrows it with a second mask (UK only, via a Pandas column converted to an array), then uses fancy indexing to pull out the ten biggest lines. Finally it simulates 1,000 possible monthly revenues by resampling real daily totals with `rng.choice`, which is the idea behind the bootstrap you will meet in Module 05.

## 8. Where it breaks

- **`and` / `or` do not work on arrays.** `(a > 1) and (a < 5)` raises "truth value of an array is ambiguous". Use `&` and `|`, and put brackets around each condition because `&` binds tighter than `>`.
- **Views bite.** `b = a[:3]; b[0] = 0` changes `a[0]`. If that is not what you want, `b = a[:3].copy()`.
- **Mask shape must match.** A mask of length 5 on an array of length 6 raises `IndexError`.
- **Broadcasting can silently do the wrong thing.** `(3, 4) × (4,)` and `(4, 3) × (4,)` both might run (the second fails, actually, but `(4, 4) × (4,)` always runs and you may have meant columns, not rows). When in doubt, print shapes.
- **Seeds give reproducibility, not randomness quality.** Use one `default_rng(seed)` per notebook and pass it around; do not re-seed inside loops or you get the same "random" draw repeatedly.
- **Legacy `np.random.seed` / `np.random.rand`** still exist and appear in old tutorials. Prefer the `default_rng` generator; it is what the rest of this course uses.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand then in code: from `rev`, select the lines that are above 16 **and** below 22. How many are there?
2. Predict the shape, then run: `np.ones((2, 3, 4)) * np.arange(4)`, `np.ones((2, 3, 4)) * np.ones((3, 1))`, `np.ones((2, 3)) * np.ones((2,))`. One of them fails; say which and why before running.
3. ShopWave stocks 5 products with prices `[2.55, 3.39, 2.75, 4.95, 1.25]`; a customer orders quantities `[6, 0, 8, 2, 0]`. Using one mask, list the prices of the products they actually ordered, and the revenue of those lines only.
4. Real data: what fraction of order lines are returns (negative quantity)? What is the total value of returns in the UK?

**Quiz**

1. What does `a[-2:]` return?
2. Why does `rev[rev > 20]` give a 1-D result even if `rev` is 2-D?
3. State the broadcasting rule in one sentence.
4. Can shapes `(5, 3)` and `(5,)` be broadcast together? What about `(5, 3)` and `(3,)`?
5. What does `np.random.default_rng(0)` guarantee, and what does it not guarantee?

## 10. Recap

- Index by position (`a[i]`, `a[r, c]`), slice by range (`a[1:4]`); slices are views, `.copy()` when needed.
- A boolean mask is a condition evaluated on the whole array; `a[mask]` selects, `mask.sum()` counts, `mask.mean()` gives the fraction. Combine with `&`, `|`, `~` and brackets.
- Broadcasting: compare shapes from the right; each pair must be equal or contain a 1. Turn a row into a column with `[:, None]`.
- `rng = np.random.default_rng(seed)` gives reproducible random numbers; every split and simulation in this course uses one.
- These three moves (select, broadcast, sample) are the mechanics underneath every ML library.

**Next:** [01.3 Pandas: Series, DataFrames, reading files, inspecting data](../03-pandas-basics/): arrays with names, which is what real data looks like.
