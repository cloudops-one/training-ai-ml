# 02.4 · Gradient descent: how a computer finds the line by walking downhill

⏱ about 2 hours · Prerequisites: 02.2, 02.3 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Lesson 02.2 gave you a formula for the best line. It works because a straight line with one feature is simple enough that calculus can solve it in one step. Most models are not that simple. A neural network has millions of knobs and no formula. XGBoost has none. Even logistic regression (Module 03) has none.

What they all have is a **loss**: a number (like MSE) that says how wrong the current knobs are. And there is one method that can turn "a number that says how wrong you are" into "better knobs", for any model: **gradient descent**. It is the engine inside almost everything from Module 03 onwards, so you will build it here, by hand, on the five customers, where you already know the answer is $m = 12.5,\ c = 25$.

## 2. The idea

You are standing on a hillside in thick fog. You want the lowest point in the valley. You cannot see it, but you can feel the slope under your feet. So:

1. Feel which way is downhill.
2. Take a step that way.
3. Repeat.

The hillside is the **loss surface**: for every possible pair of knobs $(m, c)$, the height is MSE. The slope under your feet is the **gradient**. The step size is the **learning rate**. The lowest point is the least-squares line.

```
 MSE
  ▲
  │ ╲                               ╱
  │  ╲         you start here      ╱
  │   ╲          ●                ╱
  │    ╲          ╲  step        ╱
  │     ╲          ●            ╱
  │      ╲          ╲ step     ╱
  │       ╲          ●        ╱
  │        ╲___________●_____╱   ← bottom: m = 12.5
  └────────────────────────────────▶ m
```

With two knobs the picture is a bowl, not a curve, but the rule is the same: measure the slope in each direction, step against it.

## 3. Definition

- **Loss function** $J(\theta)$: a function of the parameters that is small when predictions are good. Here, MSE as a function of $(m, c)$.
- **Gradient** $\nabla J$: the list of partial derivatives, one per parameter: how much the loss changes if you nudge *that* knob a little, holding the others fixed. It points **uphill**.
- **Learning rate** $\alpha$ (alpha): how big a step to take. Too small: thousands of steps. Too big: you overshoot the valley and climb the other side.
- **Gradient descent**: repeat $\theta \leftarrow \theta - \alpha \nabla J(\theta)$ until the loss stops falling.
- **Iteration / epoch**: one update of all the parameters.
- **Convergence**: the point where further steps change nothing worth noticing.

## 4. Formula

The loss, for the line $\hat{y}_i = m x_i + c$:

$$J(m, c) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (m x_i + c))^2$$

Its two partial derivatives (the chain rule applied once; derivation in the notebook's fold-out):

$$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n} x_i\,(y_i - \hat{y}_i)
\qquad\qquad
\frac{\partial J}{\partial c} = -\frac{2}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)$$

In words: the slope of the loss with respect to $c$ is minus twice the *average residual*; with respect to $m$ it is minus twice the *average residual weighted by $x$*. If predictions are too low on average, the residuals are positive, the gradient is negative, and the update pushes $c$ up. Exactly the fix you would make by eye.

The update rule:

$$m \leftarrow m - \alpha\,\frac{\partial J}{\partial m} \qquad\qquad c \leftarrow c - \alpha\,\frac{\partial J}{\partial c}$$

## 5. By hand

Start at $m = 0,\ c = 0$ (predict £0 for everyone), learning rate $\alpha = 0.01$.

**Iteration 1.** Predictions are all 0, so residuals $= y$: 50, 80, 90, 130, 150.

| $x$ | $y$ | $\hat{y}$ | $r = y-\hat{y}$ | $x \cdot r$ |
|---|---|---|---|---|
| 2 | 50 | 0 | 50 | 100 |
| 4 | 80 | 0 | 80 | 320 |
| 6 | 90 | 0 | 90 | 540 |
| 8 | 130 | 0 | 130 | 1040 |
| 10 | 150 | 0 | 150 | 1500 |
| | | **sum** | **500** | **3500** |

$\partial J/\partial m = -\frac{2}{5}\cdot 3500 = -1400$, $\quad \partial J/\partial c = -\frac{2}{5}\cdot 500 = -200$.

$m = 0 - 0.01\cdot(-1400) = \mathbf{14}$, $\quad c = 0 - 0.01\cdot(-200) = \mathbf{2}$.

Loss went from $J = 11280$ (all predictions 0) to $J = 244$ (line $14x + 2$). One step.

**Iteration 2.** $\hat{y} = 14x + 2$: 30, 58, 86, 114, 142. Residuals: 20, 22, 4, 16, 8 (sum 70). $x \cdot r$: 40, 88, 24, 128, 80 (sum 360).

$\partial J/\partial m = -\frac{2}{5}\cdot 360 = -144$, $\quad \partial J/\partial c = -\frac{2}{5}\cdot 70 = -28$.

$m = 14 + 1.44 = \mathbf{15.44}$, $\quad c = 2 + 0.28 = \mathbf{2.28}$. Loss $J = 125.0$.

**Iteration 3.** $\hat{y} = 15.44x + 2.28$: 33.16, 64.04, 94.92, 125.80, 156.68. Residuals: 16.84, 15.96, −4.92, 4.20, −6.68 (sum 25.4). $x \cdot r$: 33.68, 63.84, −29.52, 33.60, −66.80 (sum 34.8).

$\partial J/\partial m = -13.92$, $\quad \partial J/\partial c = -10.16$.

$m = \mathbf{15.58}$, $\quad c = \mathbf{2.38}$. Loss $J = 121.3$.

Notice what happened: $m$ overshot to 15.6 (the answer is 12.5) while $c$ crawls from 2 towards 25. The $m$ gradient is large because it is weighted by $x$; the $c$ gradient is small. The walk will now slowly trade $m$ down and $c$ up along the valley floor. In the notebook you watch it take about **5,000 steps** to arrive at $(12.5, 25)$. That slowness has a name, and a fix, in Module 04 (feature scaling).

## 6. In code

[lesson.ipynb](lesson.ipynb) is the whole algorithm in ten lines:

```python
m, c, alpha = 0.0, 0.0, 0.01
for step in range(5000):
    y_hat = m * x + c
    resid = y - y_hat
    grad_m = -2 * (x * resid).mean()
    grad_c = -2 * resid.mean()
    m -= alpha * grad_m
    c -= alpha * grad_c
print(m, c)       # → 12.5, 25.0 (to 2 decimals)
```

Then it plots: the loss falling step by step; the path of $(m, c)$ across the bowl; and what happens with $\alpha = 0.001$ (crawls), $\alpha = 0.01$ (arrives), $\alpha = 0.03$ (diverges: the loss explodes to infinity in a few steps).

## 7. On real data

Gradient descent on the 3,317-customer table, past spend → next-quarter spend, reaches the same line scikit-learn found in 02.3 (slope ≈ 0.54). But only after the feature is **scaled**: raw spend ranges from £3 to £280,000, and with such numbers the gradient for the slope is millions of times bigger than for the intercept; no single learning rate works for both. The notebook shows the divergence on raw data and the fix (divide by the standard deviation), which is the preview of lesson 04.5.

## 8. Where it breaks

- **Learning rate too big**: overshoots, loss grows, numbers become `inf`/`nan`. The most common failure in all of deep learning. Fix: smaller $\alpha$.
- **Learning rate too small**: converges, eventually, after a very long time. Fix: bigger $\alpha$, or a schedule that shrinks it over time.
- **Unscaled features**: one knob's gradient dwarfs the others'; the walk zigzags down a long narrow canyon. Fix: scale features (04.5).
- **Local minima**: MSE for a line is one bowl with one bottom. Neural-network losses are mountain ranges with many valleys; gradient descent finds *a* valley, not *the* valley. Module 08.
- **Computing the gradient over all rows** every step is slow for millions of rows. Fix: **stochastic** gradient descent uses a random mini-batch per step (08.6).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: start at $m = 10,\ c = 30$ with $\alpha = 0.01$. Do one iteration. Does $m$ go up or down? Does $c$? Explain the direction from the sign of the residuals.
2. In code: run gradient descent with $\alpha \in \{0.001, 0.005, 0.01, 0.02, 0.025, 0.03\}$ for 2,000 steps each. Tabulate the final loss. Where is the edge between converging and diverging?
3. Add a stopping rule: stop when the loss improves by less than $10^{-6}$ between steps. How many steps does $\alpha = 0.01$ need?
4. Scale $x$ by subtracting its mean and dividing by its standard deviation, run gradient descent with $\alpha = 0.1$, and convert the learned slope back to original units. How many steps to converge now?

**Quiz**

1. Which direction does the gradient point: uphill or downhill?
2. In the update rule, why is there a minus sign?
3. What goes wrong if the learning rate is too large? Too small?
4. After iteration 1, $m$ jumped to 14 while $c$ only reached 2. Why the difference?
5. Lesson 02.2 solved the same problem with a formula. Why learn gradient descent at all?

## 10. Recap

- A **loss** measures how wrong the knobs are; the **gradient** says which way makes it worse; step the other way.
- $\theta \leftarrow \theta - \alpha \nabla J$: the one update rule behind linear regression, logistic regression, boosting (in spirit) and every neural network.
- For MSE and a line: $\partial J/\partial m = -\frac{2}{n}\sum x_i r_i$, $\ \partial J/\partial c = -\frac{2}{n}\sum r_i$, with $r_i$ the residuals.
- Three hand iterations took the loss from 11,280 to 121; a few thousand more reach (12.5, 25).
- Learning rate and feature scale decide whether it converges at all.

**Next:** [02.5 Many features](../05-multiple-regression/) uses more than one column, and the formula from 02.2 and the loop from here both generalise without changing shape.
