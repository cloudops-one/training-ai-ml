# 08.4 · Backpropagation on a two-layer network, by hand

⏱ about 3 hours · Prerequisites: 08.2, 08.3, 02.4 (gradient descent) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

In lesson 02.4 gradient descent fitted a line by nudging two numbers, the slope and the intercept, in the direction that lowered the error. You could write the two derivatives by hand.

A network has many more numbers. The small churn network in lesson 08.5 has 241; image networks have millions. Each weight sits inside layers of other weights and activations. How do you find out which way to nudge a weight in the first layer, when its effect on the loss passes through every layer after it?

**Backpropagation** answers that question. It is not a new learning method; it is an efficient way to compute every derivative gradient descent needs, by passing blame backwards from the loss, one layer at a time. Once you have done it once with numbers, "the network learns" stops being magic.

## 2. The idea

Picture a relay team that lost a race by 2 seconds. To work out who should train harder, you go **backwards**: the last runner lost 1 second on their leg; they started 1 second behind because the third runner was slow, and so on. Each runner's share of the blame depends on the runners after them.

Backpropagation does the same with the loss:

1. **Forward pass**: compute every neuron's output, layer by layer, and the loss.
2. **Backward pass**: start from the loss and ask "how much does the loss change if this number changes?" for the output, then the last layer's weights, then the hidden outputs, then the first layer's weights. Each step reuses the answer from the step after it.
3. **Update**: move every weight a little against its derivative (gradient descent, lesson 02.4).

Repeat thousands of times and the network learns.

## 3. Definition

- **Gradient**: the derivative of the loss with respect to a parameter, $\partial L / \partial w$. Positive means "increasing $w$ increases the loss", so we decrease $w$.
- **Chain rule**: if $L$ depends on $a$, and $a$ depends on $w$, then $\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial a} \cdot \dfrac{\partial a}{\partial w}$. Slopes along a path multiply.
- **Backpropagation**: applying the chain rule from the loss back to every weight, reusing shared pieces so each derivative is computed once.
- **Learning rate** $\eta$: how big a step to take against the gradient (lesson 02.4).
- **Automatic differentiation**: libraries such as TensorFlow and PyTorch record the forward pass and run backpropagation for you. `tf.GradientTape` is TensorFlow's recorder.

## 4. Formula

The network: 2 inputs → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid), trained with binary cross-entropy.

```
x₁ ──┬── w₁₁ ──► [ h₁ ] ──── v₁ ──┐
     └── w₁₂ ──┐                   ├──► [ ŷ ] ──► loss L
x₂ ──┬── w₂₁ ──┘► [ h₂ ] ──── v₂ ──┘
     └── w₂₂ ───►
```

$w_{ij}$ connects input $i$ to hidden neuron $j$; $v_j$ connects hidden neuron $j$ to the output.

**Forward**

$$z_j = x_1 w_{1j} + x_2 w_{2j} + b_j, \quad h_j = \sigma(z_j), \quad z_{\text{out}} = h_1 v_1 + h_2 v_2 + c, \quad \hat y = \sigma(z_{\text{out}})$$

$$L = -\big[y \ln \hat y + (1-y)\ln(1-\hat y)\big]$$

**Backward**, each line using the one above it:

| Derivative | Formula | Why |
|---|---|---|
| output | $\delta_{\text{out}} = \dfrac{\partial L}{\partial z_{\text{out}}} = \hat y - y$ | log loss and sigmoid simplify beautifully (see below) |
| output weights | $\dfrac{\partial L}{\partial v_j} = \delta_{\text{out}} \, h_j$, $\quad \dfrac{\partial L}{\partial c} = \delta_{\text{out}}$ | $z_{\text{out}}$ grows by $h_j$ per unit of $v_j$ |
| hidden outputs | $\dfrac{\partial L}{\partial h_j} = \delta_{\text{out}} \, v_j$ | $z_{\text{out}}$ grows by $v_j$ per unit of $h_j$ |
| hidden pre-activations | $\delta_j = \dfrac{\partial L}{\partial h_j} \, h_j(1-h_j)$ | sigmoid's slope is $\sigma(z)(1-\sigma(z))$ |
| input weights | $\dfrac{\partial L}{\partial w_{ij}} = \delta_j \, x_i$, $\quad \dfrac{\partial L}{\partial b_j} = \delta_j$ | $z_j$ grows by $x_i$ per unit of $w_{ij}$ |

**Update** every parameter: $w \leftarrow w - \eta \, \dfrac{\partial L}{\partial w}$.

*Why $\hat y - y$?* The loss's slope with respect to $\hat y$ is $-\dfrac{y}{\hat y} + \dfrac{1-y}{1-\hat y}$; the sigmoid's slope is $\hat y(1-\hat y)$. Multiply them and everything cancels to $\hat y - y$: *prediction minus truth*. That is why sigmoid outputs and binary cross-entropy are always used together.

## 5. By hand

One ShopWave customer with two scaled features $x = (1.0,\ 0.5)$ who **churned** ($y = 1$). Starting weights:

$$w_{11} = 0.2,\ w_{12} = 0.6,\ w_{21} = -0.4,\ w_{22} = 0.1,\quad b_1 = b_2 = 0,\quad v_1 = 0.5,\ v_2 = -0.3,\ c = 0,\quad \eta = 0.5$$

**Forward pass**

| Step | Calculation | Value |
|---|---|---|
| $z_1$ | $1.0(0.2) + 0.5(-0.4) + 0$ | 0.000 |
| $h_1$ | $\sigma(0)$ | 0.5000 |
| $z_2$ | $1.0(0.6) + 0.5(0.1) + 0$ | 0.650 |
| $h_2$ | $\sigma(0.65)$ | 0.6570 |
| $z_{\text{out}}$ | $0.5(0.5) + 0.657(-0.3) + 0$ | 0.0529 |
| $\hat y$ | $\sigma(0.0529)$ | 0.5132 |
| $L$ | $-\ln 0.5132$ | **0.667** |

The network is barely better than a coin toss for a customer who churned.

**Backward pass**

| Derivative | Calculation | Value |
|---|---|---|
| $\delta_{\text{out}}$ | $0.5132 - 1$ | −0.4868 |
| $\partial L / \partial v_1$ | $-0.4868 \times 0.5$ | −0.2434 |
| $\partial L / \partial v_2$ | $-0.4868 \times 0.657$ | −0.3198 |
| $\partial L / \partial c$ | $\delta_{\text{out}}$ | −0.4868 |
| $\partial L / \partial h_1$ | $-0.4868 \times 0.5$ | −0.2434 |
| $\partial L / \partial h_2$ | $-0.4868 \times (-0.3)$ | +0.1460 |
| $\delta_1$ | $-0.2434 \times 0.5 \times 0.5$ | −0.0608 |
| $\delta_2$ | $0.1460 \times 0.657 \times 0.343$ | +0.0329 |
| $\partial L / \partial w_{11}$ | $\delta_1 \times x_1 = -0.0608 \times 1.0$ | −0.0608 |
| $\partial L / \partial w_{21}$ | $\delta_1 \times x_2 = -0.0608 \times 0.5$ | −0.0304 |
| $\partial L / \partial w_{12}$ | $\delta_2 \times x_1 = 0.0329 \times 1.0$ | +0.0329 |
| $\partial L / \partial w_{22}$ | $\delta_2 \times x_2 = 0.0329 \times 0.5$ | +0.0165 |

Read it like a story. The output was too low ($\delta_{\text{out}} < 0$), so every weight that feeds it a positive signal should grow: both $v$'s get negative gradients and will increase. Hidden neuron 2 currently *pulls the output down* ($v_2 = -0.3$), so its own activity should shrink: its input weights get positive gradients and will decrease.

**Update** ($\eta = 0.5$, so subtract half of each gradient)

| Parameter | Old | New |
|---|---|---|
| $v_1$ | 0.5 | $0.5 - 0.5(-0.2434) = 0.6217$ |
| $v_2$ | −0.3 | $-0.3 - 0.5(-0.3198) = -0.1401$ |
| $c$ | 0 | 0.2434 |
| $w_{11}$ | 0.2 | 0.2304 |
| $w_{21}$ | −0.4 | −0.3848 |
| $w_{12}$ | 0.6 | 0.5835 |
| $w_{22}$ | 0.1 | 0.0918 |
| $b_1$, $b_2$ | 0, 0 | 0.0304, −0.0165 |

**Forward again** with the new weights: $\hat y = 0.6163$, loss $= -\ln 0.6163 = $ **0.484**, down from 0.667. One step of backpropagation made the network noticeably more confident about the right answer.

## 6. In code

The notebook ([lesson.ipynb](lesson.ipynb)) does three things:

1. Re-computes every row of the three tables above in NumPy, with the same variable names.
2. Wraps forward, backward and update into a function and trains the same 2-2-1 network on **XOR**, the pattern a single neuron cannot learn (lesson 08.2): output 1 when exactly one input is 1. After 1,000 steps it predicts 0.009, 0.992, 0.989 and 0.007 for the four cases (targets 0, 1, 1, 0).
3. Asks TensorFlow for the same gradients with `tf.GradientTape`, and prints them next to the hand-computed ones. They agree to the last digit.

```python
with tf.GradientTape() as tape:
    loss = forward(x, y, params)            # TensorFlow records every operation
grads = tape.gradient(loss, params)         # ...then runs backpropagation for you
```

From lesson 08.5 on, Keras calls this for you inside `model.fit()`.

## 7. On real data

The notebook shows that the hidden layer is what makes XOR possible. It plots the network's decision boundary after training: two straight lines (one per hidden neuron) combined into a band, which no single neuron can draw.

It also shows a run that **stalls**: with a different random starting point (seed 5) the same network is still predicting about 0.34 for three of the four cases after 1,000 steps. It has landed on a **plateau**: a nearly flat stretch of the loss surface where the gradients are tiny, so each step barely moves it. Given 5,000 steps, a larger learning rate or two more hidden neurons, it escapes and learns XOR perfectly (exercise 4). Real networks avoid most plateaus by being wider than strictly necessary and by using optimisers with momentum (lesson 08.6).

## 8. Where it breaks

- **Vanishing gradients.** Each sigmoid layer multiplies the backward signal by at most 0.25. Ten sigmoid layers can shrink it a millionfold, so early layers stop learning. ReLU and careful initialisation fix most of this (lesson 08.3).
- **Exploding gradients.** The opposite: large weights multiply the signal up until the updates overshoot. Smaller learning rates and gradient clipping help.
- **Plateaus and bad starts.** As seed 5 shows, a run can crawl for a long time where the loss surface is flat, and in some problems it can settle in a genuinely poor valley. Wider networks and momentum-based optimisers make both much rarer.
- **Getting the maths right by hand does not scale.** Nobody derives gradients for a million weights by hand. Libraries do it automatically and reliably; your job is to understand what they compute, so you can recognise when training goes wrong.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: redo the forward pass with the *updated* weights and confirm $\hat y = 0.6163$ and $L = 0.484$.
2. In code: take a second update step. Does the loss keep falling? Take 50 steps and plot it.
3. Repeat the by-hand example for a customer who **stayed** ($y = 0$) with the same inputs and starting weights. Which weights move the other way?
4. XOR with seed 5 stalls. Try (a) more steps, (b) a larger learning rate, (c) 4 hidden neurons instead of 2. Which fixes it?

**Quiz** (answers at the end of the notebook)

1. What does the forward pass compute, and what does the backward pass compute?
2. Why is it called *back*propagation?
3. For a sigmoid output with binary cross-entropy, what is $\partial L / \partial z_{\text{out}}$?
4. Why did hidden neuron 2's input weights *decrease* in the update?
5. What does `tf.GradientTape` do?

## 10. Recap

- Backpropagation computes every gradient with the chain rule, starting at the loss and reusing each layer's result for the layer before it.
- For sigmoid + log loss the output gradient is simply $\hat y - y$; each earlier gradient multiplies in a local slope ($v_j$, $h_j(1-h_j)$, $x_i$).
- By hand: one update moved the prediction for a churner from 0.513 to 0.616 and cut the loss from 0.667 to 0.484.
- A 2-2-1 network learns XOR in 1,000 steps; a single neuron never can. Unlucky starting weights can leave it crawling across a plateau.
- TensorFlow's `GradientTape` produces exactly the hand-computed gradients. From now on, the library does this part.

**Next:** [08.5 Your first network in Keras](../05-first-keras-network/).
