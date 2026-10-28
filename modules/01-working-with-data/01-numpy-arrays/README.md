# 01.1 · NumPy: arrays, shapes and why vectors beat loops

⏱ about 2 hours · Prerequisites: Python lists and loops · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Finance asks for the revenue of every order line in 2011: quantity times unit price, for 541,910 rows. You know how to do this in Python:

```python
revenue = []
for q, p in zip(quantities, prices):
    revenue.append(q * p)
```

It works. It also takes a noticeable fraction of a second, and finance wants it per country, per month, per product, with tax, with discounts, re-run every morning. Then in Module 02 you will train a model that does this kind of arithmetic *millions* of times while it learns. A plain Python loop is a hundred times too slow for that, and the code to do it on a 2-D table of numbers becomes a nest of loops nobody wants to read.

Every machine-learning library you will meet (Pandas, scikit-learn, TensorFlow, PyTorch) is built on one idea: **do arithmetic on whole blocks of numbers at once**. The block is called an array. The library is NumPy. This lesson is about thinking in arrays instead of loops.

## 2. The idea

A Python list is a row of boxes, each holding *anything*: an integer, a string, another list. Python has to check what is in every box before it can add two of them. That is flexible and slow.

A NumPy **array** is a row of boxes that all hold the *same kind* of number, packed side by side in memory. Because NumPy knows in advance they are all, say, 64-bit floats, it can multiply two arrays with one tight loop written in C, with no checking. That is rigid and fast.

```
Python list:   [ 6 ] [ 6 ] [ 8 ] [ 6 ] [ 6 ]      each box: "what type am I?"
                 ↓     ↓     ↓     ↓     ↓          then multiply, one at a time

NumPy array:   | 6 | 6 | 8 | 6 | 6 |              one type for all, one instruction:
                       × prices                      "multiply these 5 floats"
               |2.55|3.39|2.75|3.39|3.39|
```

The second idea is **shape**. A 1-D array is a list of numbers. A 2-D array is a grid: rows and columns, like a spreadsheet. A 3-D array is a stack of grids (a colour image is one: height × width × 3 colours). Every array knows its shape, and arithmetic between arrays is defined in terms of shapes.

## 3. Definition

An **ndarray** ("n-dimensional array") is a grid of values, all of the same type, with these attributes:

- **`dtype`**: the type of every element (`int64`, `float64`, `bool`, …). One type for the whole array.
- **`shape`**: a tuple with the length of each dimension. `(5,)` is five numbers in a row; `(3, 4)` is 3 rows × 4 columns; `(28, 28, 3)` is an image.
- **`ndim`**: the number of dimensions, which is `len(shape)`.
- **`size`**: total number of elements, the product of the shape.

**Vectorised operation**: an operation applied to every element of an array (or to matching elements of two arrays) in one call, without a Python loop. `a * b`, `a + 1`, `np.sqrt(a)`, `a.sum()` are all vectorised.

## 4. Formula

Element-wise arithmetic between two arrays of the same shape:

$$(a \times b)_i = a_i \times b_i \quad \text{for every position } i$$

A scalar is applied to every element:

$$(a \times 1.2)_i = a_i \times 1.2$$

An aggregation collapses the array to one number:

$$\text{sum}(a) = \sum_i a_i \qquad \text{mean}(a) = \frac{1}{n}\sum_i a_i$$

For a 2-D array of shape $(r, c)$, `sum(axis=0)` adds *down* each column and returns shape $(c,)$; `sum(axis=1)` adds *across* each row and returns shape $(r,)$. Remember: **the axis you name is the one that disappears.**

## 5. By hand

Five order lines from ShopWave's first invoice of December 2010:

| line | quantity | price (£) |
|---|---|---|
| 1 | 6 | 2.55 |
| 2 | 6 | 3.39 |
| 3 | 8 | 2.75 |
| 4 | 6 | 3.39 |
| 5 | 6 | 3.39 |

**Element-wise product** (line revenue):

| line | quantity × price | revenue |
|---|---|---|
| 1 | 6 × 2.55 | 15.30 |
| 2 | 6 × 3.39 | 20.34 |
| 3 | 8 × 2.75 | 22.00 |
| 4 | 6 × 3.39 | 20.34 |
| 5 | 6 × 3.39 | 20.34 |

**Sum**: 15.30 + 20.34 + 22.00 + 20.34 + 20.34 = **98.32** (invoice total).

**Mean** quantity: (6 + 6 + 8 + 6 + 6) / 5 = 32 / 5 = **6.4**.

**Shapes**: quantities have shape `(5,)`, prices `(5,)`, their product `(5,)`, the sum `()` (a single number). Stack quantity and price side by side and you get a `(5, 2)` grid: 5 rows, 2 columns. `sum(axis=0)` on that grid gives `(2,)`: total quantity 32 and total price 15.47. `sum(axis=1)` gives `(5,)`: quantity + price per line, which is meaningless here, which is the point: **axis is a choice you make on purpose**.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Build the five-line arrays and check `shape`, `dtype`, `ndim`, `size`.
2. Compute revenue with a loop, then with `*`, and see the same 98.32.
3. Build a `(5, 2)` grid and watch `axis=0` vs `axis=1`.
4. Create arrays the ways you will actually use: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`.
5. Time a loop against the vectorised version on 541,910 real rows.

```python
import numpy as np
qty = np.array([6, 6, 8, 6, 6])
price = np.array([2.55, 3.39, 2.75, 3.39, 3.39])
revenue = qty * price          # one line, no loop
revenue.sum()                  # 98.32
```

## 7. On real data

The notebook loads the 2010–2011 Online Retail sheet, pulls `Quantity` and `Price` into NumPy arrays and computes line revenue three ways: a Python loop, a list comprehension, and `qty * price`. You will see the timings side by side. Then you sum, take the mean, and find the biggest single line. From here on in the course, if you catch yourself writing `for` over rows of numbers, stop and look for the array version.

## 8. Where it breaks

- **Mixed types collapse to the slowest.** `np.array([1, 2.5])` becomes float. `np.array([1, "a"])` becomes strings, and `* 2` will fail. Keep arrays numeric; keep text in Pandas (next lessons).
- **Integer overflow.** `np.int8` holds −128…127; `np.array([100], dtype=np.int8) * 2` is **−56**, silently. Default `int64` is fine for this course; just know the trap exists.
- **Shape mismatch.** `np.array([1, 2, 3]) * np.array([1, 2])` raises `ValueError: operands could not be broadcast`. Lesson 01.2 explains exactly when shapes are compatible.
- **Loops are not always wrong.** For 5 elements, a loop is fine and clearer. Vectorise when the data is big or the operation will run many times.
- **Copies vs views.** Slicing an array gives a *view* of the same memory; changing the slice changes the original. Covered in 01.2; be warned now.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand, then in code: ShopWave adds 20% VAT. Compute the VAT-inclusive revenue of the five lines and the new invoice total.
2. Make a `(3, 4)` array of ones, multiply it by `np.arange(4)`, and predict the result of `sum(axis=0)` and `sum(axis=1)` *before* running.
3. Write `mean` without `.mean()`: using only `.sum()` and `.size`. Check it against `np.mean` on the quantities.
4. On the real data: how many order lines have revenue above £1,000? (Hint: `(revenue > 1000).sum()` works because `True` counts as 1. Lesson 01.2 explains why.)

**Quiz**

1. Why can NumPy multiply two arrays faster than Python multiplies two lists?
2. What is the shape of a 2-D array with 7 rows and 3 columns, and what does `sum(axis=1)` return?
3. What `dtype` does `np.array([1, 2, 3.0])` get, and why?
4. `a.size` is 24 and `a.ndim` is 3. Give one possible `a.shape`.
5. Name one case where a plain Python loop is the better choice.

## 10. Recap

- A NumPy array is a block of same-typed numbers with a `shape`; arithmetic applies to the whole block at once.
- `qty * price` does what a loop does, in C, roughly 100× faster, in one readable line.
- `shape` is a tuple; `ndim` is its length; `size` is its product; `dtype` is the one type shared by all elements.
- Aggregations with `axis=`: the axis you name is the one that disappears.
- Every ML library speaks arrays. Learning to think in shapes now saves hours of confusion later.

**Next:** [01.2 Indexing, masks, broadcasting and random numbers](../02-numpy-indexing-broadcasting/): how to pick out the rows you want and how arrays of *different* shapes combine.
