# 08.2 · One neuron, by hand and in NumPy

⏱ about 2 hours · Prerequisites: 08.1, 03.4 (logistic regression, sigmoid), 04.5 (scaling) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Neural networks have a reputation for being mysterious: "brain-inspired", "black boxes", "billions of parameters". It is easy to feel that you need to trust them rather than understand them.

You do not. A network of any size is built from one unit, repeated: the **neuron**. If you can compute one neuron with a pencil, and you can compute a **layer** of them with one matrix multiplication, you understand every number a network computes. The rest is repetition and a training procedure.

And you have already met a neuron. You trained one in Module 03 without knowing its name.

## 2. The idea

A neuron is a tiny judge. It looks at a few inputs, decides how much each one matters (its **weights**), adds a baseline opinion (its **bias**), and then squashes the total into a useful range with an **activation function**.

For ShopWave churn, one neuron might think: *"Many days since the last order? Raises my suspicion a lot. Many orders? Lowers it. High spend? Lowers it a little."* It adds those up and squashes the total into a probability between 0 and 1.

```
 days_since_last ──(w₁)──┐
 orders ─────────(w₂)────┼──►  Σ  + b  ──►  activation  ──►  output
 spend ──────────(w₃)────┘
```

A **layer** is many neurons looking at the same inputs, each with its own weights. A **network** is layers feeding layers.

## 3. Definition

- **Input** $x$: a vector of numbers for one example, here one customer's scaled features.
- **Weight** $w_i$: how strongly input $i$ pushes the neuron's output up (positive) or down (negative).
- **Bias** $b$: the neuron's output when every input is zero; it shifts the decision.
- **Pre-activation** $z$: the weighted sum plus bias.
- **Activation function** $f$: a fixed function applied to $z$. Sigmoid squashes into (0, 1); others come in lesson 08.3.
- **Dense (fully connected) layer**: $k$ neurons that each see every input. Keras calls it `Dense(k)`.
- **Parameters**: the weights and biases that training learns. A dense layer from $n$ inputs to $k$ neurons has $n \times k + k$ of them.

## 4. Formula

One neuron:

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = w \cdot x + b \qquad a = f(z)$$

With the sigmoid activation $f(z) = \dfrac{1}{1 + e^{-z}}$, this is **exactly logistic regression** from lesson 03.4: same weighted sum, same squash, same output, a probability.

A layer of $k$ neurons applied to $m$ examples at once:

$$H = f(XW + b)$$

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $m \times n$ | $m$ customers, $n$ features each |
| $W$ | $n \times k$ | one column of weights per neuron |
| $b$ | $k$ | one bias per neuron, added to every row |
| $H$ | $m \times k$ | every customer's output from every neuron |

The inner sizes must match ($n$ and $n$); the result takes the outer sizes ($m$ and $k$).

## 5. By hand

**One neuron, one customer.** Customer C042's features, already scaled as in lesson 04.5 (so 0 means "average" and 1 means "one standard deviation above"):

| Feature | Scaled value $x$ | Weight $w$ | $w \times x$ |
|---|---|---|---|
| days since last order | 1.5 | 0.9 | 1.35 |
| orders | −0.5 | −1.1 | 0.55 |
| spend | −0.8 | −0.4 | 0.32 |
| | | **sum** | **2.22** |

Add the bias $b = -0.3$: $z = 2.22 - 0.3 = 1.92$.

Squash with the sigmoid: $a = \dfrac{1}{1 + e^{-1.92}} = \dfrac{1}{1 + 0.1466} = 0.872$.

The neuron says C042 has an **87%** chance of churning. Read the table: long silence pushed hardest, and *below-average* orders and spend also pushed **up**, because a negative weight times a negative input is positive.

**A layer: three customers, four neurons.** Stack three customers as rows of $X$ (3 × 3) and give a layer four neurons, so $W$ is 3 × 4:

$$X = \begin{bmatrix} 1.5 & -0.5 & -0.8 \\ -0.7 & 1.2 & 0.9 \\ 0.2 & 0.0 & -0.1 \end{bmatrix}
\qquad
W = \begin{bmatrix} 0.9 & -0.2 & 0.5 & 0.1 \\ -1.1 & 0.4 & 0.3 & -0.6 \\ -0.4 & 0.7 & -0.2 & 0.8 \end{bmatrix}
\qquad b = [-0.3,\ 0.1,\ 0.0,\ 0.2]$$

Shapes: (3 × 3) times (3 × 4) gives **3 × 4**: one row per customer, one column per neuron. Entry (row 1, column 1) is exactly the neuron you just computed, before squashing: 1.92. Working out the rest:

$$XW + b = \begin{bmatrix} 1.92 & -0.96 & 0.76 & 0.01 \\ -2.61 & 1.35 & -0.17 & 0.13 \\ -0.08 & -0.01 & 0.12 & 0.14 \end{bmatrix}$$

Check one more: row 2, column 2 is $(-0.7)(-0.2) + (1.2)(0.4) + (0.9)(0.7) + 0.1 = 0.14 + 0.48 + 0.63 + 0.1 = 1.35$.

This layer has $3 \times 4 + 4 = 16$ parameters. Hidden layers usually use the ReLU activation, $\max(0, z)$, which simply zeroes the negatives (lesson 08.3).

## 6. In code

The notebook ([lesson.ipynb](lesson.ipynb)) does it three ways:

```python
z = x @ w + b                         # one neuron: dot product plus bias
a = 1 / (1 + np.exp(-z))              # sigmoid → 0.872

H = X @ W + b                         # a layer for three customers at once → shape (3, 4)
```

and then in Keras:

```python
layer = keras.layers.Dense(4)
layer.build((None, 3))
layer.set_weights([W, b])             # use our hand-picked numbers
layer(X)                              # identical to X @ W + b
```

## 7. On real data

The notebook trains scikit-learn's logistic regression on the real ShopWave churn table (five scaled features), then copies its five coefficients and intercept into a single Keras neuron, `Dense(1, activation="sigmoid")`. On all 830 test customers the two give the same probabilities, to within 0.0000002 (floating-point rounding).

So the very first network you build *is* the model you have trusted since Module 03. Everything a bigger network adds comes from stacking more neurons and layers, which lesson 08.5 does.

## 8. Where it breaks

- **One neuron draws one straight boundary.** Like logistic regression, it cannot learn "churn if *either* very new *or* long silent" or the XOR pattern. Stacking layers fixes this (lesson 08.4).
- **Unscaled inputs.** A neuron multiplies inputs by weights; spend in pounds (thousands) would swamp orders (single digits). Scale first, always (lesson 04.5).
- **Shapes.** Most beginner errors in deep learning are shape errors. Write the shapes down: $(m \times n) \cdot (n \times k) = (m \times k)$.
- **"Neurons are like brain cells."** Only loosely. The analogy helped name the idea in the 1950s; it does not help you debug.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: customer C077 has scaled features (−1.0, 2.0, 1.5). Using the weights and bias from section 5, compute $z$ and the churn probability. Check in NumPy.
2. A layer takes 5 inputs and has 16 neurons. Its output feeds a layer of 8 neurons, then 1 output neuron. Count the parameters of each layer and the total, by hand. (You will meet this exact network in 08.5.)
3. Apply ReLU to the 3 × 4 matrix from section 5 by hand. How many of the 12 outputs are exactly zero?
4. Change the bias in section 5 until C042's churn probability is 50%. What value of $b$ does that need, and why?

**Quiz** (answers at the end of the notebook)

1. What three things does a neuron compute, in order?
2. Why is a single sigmoid neuron the same model as logistic regression?
3. $X$ is 100 × 5 and the layer has 16 neurons. What are the shapes of $W$, $b$ and the output?
4. How many parameters does `Dense(10)` have when it receives 64 inputs?
5. What can a single neuron *not* learn, and what fixes it?

## 10. Recap

- A neuron computes $a = f(w \cdot x + b)$: weighted sum, plus bias, through an activation.
- With a sigmoid activation, one neuron is logistic regression. Copying the coefficients into Keras gave identical predictions.
- A layer is a matrix multiplication: $(m \times n)(n \times k) + b = (m \times k)$, with $n k + k$ parameters.
- By hand: C042 scored $z = 1.92$, a churn probability of 0.872.
- One neuron draws one straight boundary; layers are needed for anything more.

**Next:** [08.3 Activation functions and loss functions](../03-activations-and-losses/).
