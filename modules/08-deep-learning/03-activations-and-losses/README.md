# 08.3 · Activation functions and loss functions

⏱ about 1.5 hours · Prerequisites: 08.2, 02.3 (MSE), 03.4 (log loss) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Lesson 08.2 ended with a limit: one neuron draws one straight boundary. The obvious fix is to stack more layers. But if you stack layers of plain weighted sums, something disappointing happens: the whole stack is *still* one straight boundary. Three layers, sixteen neurons each, and the result is exactly as good as logistic regression.

Two small choices decide whether a network can learn anything interesting, and whether it learns the right thing:

- the **activation function** each neuron applies, which gives the network its bends;
- the **loss function** training minimises, which tells it what "wrong" means for your task.

Get either wrong and the network trains happily towards nothing useful.

## 2. The idea

**Activations are bends.** A weighted sum is a flat surface. Feeding a flat surface into another weighted sum gives another flat surface. Bend it in between, even with something as crude as "replace negatives with zero", and stacked layers can build curves, corners and islands out of many small bends. That is how a network learns the moon-shaped boundary in the notebook.

**Losses are the scorecard.** For spend predictions you care how many pounds off you were, so you square the errors (MSE, Module 02). For churn you predicted a probability, and the scorecard should punish *confident wrong* answers hardest (log loss, Module 03). For "which of 10 product types" you need a probability for each type, and the loss looks only at the probability given to the right one.

## 3. Definition

**Activation functions** (applied to each neuron's $z$):

- **Sigmoid** $\sigma(z) = \dfrac{1}{1+e^{-z}}$: squashes to (0, 1). Used for a **binary output** (a probability).
- **Tanh** $\tanh(z) = \dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$: squashes to (−1, 1), centred on zero. Common inside recurrent networks (08.10).
- **ReLU** $\max(0, z)$: zero for negatives, unchanged for positives. The default for **hidden layers**: cheap, and its slope does not fade for large $z$.
- **Softmax** $\text{softmax}(z)_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$: turns a vector of scores into probabilities that sum to 1. Used for a **multi-class output**.
- **Linear** (no activation): $f(z) = z$. Used for a **regression output** (any number).

**Loss functions** (one number: how wrong the network is on a batch):

- **Mean squared error** $\text{MSE} = \frac{1}{n}\sum (y - \hat y)^2$: regression.
- **Binary cross-entropy (log loss)** $-\big[y \ln \hat p + (1-y)\ln(1-\hat p)\big]$: one probability, two classes.
- **Categorical cross-entropy** $-\ln \hat p_{\text{true class}}$: several classes. Keras calls it `sparse_categorical_crossentropy` when the labels are integers (0, 1, 2…) and `categorical_crossentropy` when they are one-hot vectors.

**Logit**: the raw score $z$ before the final sigmoid or softmax.

## 4. Formula

Why stacking linear layers collapses. Two layers with no activation:

$$h = x W_1, \qquad \hat y = h W_2 = (x W_1) W_2 = x (W_1 W_2)$$

$W_1 W_2$ is just another matrix. Two layers are one layer in disguise, and so are a hundred. Put any non-linear $f$ in between, $\hat y = f(x W_1) W_2$, and the shortcut no longer exists.

Which output and which loss:

| Task | Example | Output layer | Loss |
|---|---|---|---|
| Regression | next month's spend | `Dense(1)` (linear) | MSE (or MAE) |
| Binary classification | churn yes/no | `Dense(1, activation="sigmoid")` | binary cross-entropy |
| Multi-class, one label | product type | `Dense(k, activation="softmax")` | categorical cross-entropy |
| Multi-label | a photo can be "red" *and* "mug" | `Dense(k, activation="sigmoid")` | binary cross-entropy |

## 5. By hand

**Activations at three points.**

| $z$ | sigmoid | tanh | ReLU |
|---|---|---|---|
| −2 | $1/(1+e^{2}) = 0.119$ | −0.964 | 0 |
| 0 | 0.5 | 0 | 0 |
| 2 | $1/(1+e^{-2}) = 0.881$ | 0.964 | 2 |

**Softmax of three product-type scores** $z = [2.0, 1.0, 0.1]$ for (mug, cushion, lantern):

$e^{2.0} = 7.389$, $e^{1.0} = 2.718$, $e^{0.1} = 1.105$, total $11.213$.

Probabilities: $7.389/11.213 = 0.659$, $2.718/11.213 = 0.242$, $1.105/11.213 = 0.099$. They sum to 1, and the biggest score keeps the biggest share.

**Linear layers collapse.** Input $x = [1, 3]$, weights

$$W_1 = \begin{bmatrix} 1 & 2 \\ 0.5 & -1 \end{bmatrix}, \qquad W_2 = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$

- Two layers: $h = xW_1 = [1 + 1.5,\ 2 - 3] = [2.5,\ -1]$, then $hW_2 = 7.5 - 1 = 6.5$.
- One layer: $W_1W_2 = [1\cdot3 + 2\cdot1,\ 0.5\cdot3 - 1\cdot1] = [5,\ 0.5]$, then $x \cdot [5, 0.5] = 5 + 1.5 = 6.5$. **Same answer.**
- With ReLU between: $\text{ReLU}(h) = [2.5,\ 0]$, then $7.5 + 0 = 7.5$. **Different**: the bend changed what the network can say.

**Losses.**

- MSE: true spends [3, 5], predictions [2.5, 6]: $\big((0.5)^2 + (1)^2\big)/2 = (0.25 + 1)/2 = 0.625$.
- Binary cross-entropy, customer who churned ($y = 1$): predicted 0.9 → $-\ln 0.9 = 0.105$. Predicted 0.1 → $-\ln 0.1 = 2.303$, twenty-two times worse.
- Categorical cross-entropy, true class "cushion", predicted [0.2, 0.7, 0.1]: $-\ln 0.7 = 0.357$. The probabilities given to the wrong classes do not enter the formula directly; they only matter by taking probability away from the right one.

## 6. In code

The notebook ([lesson.ipynb](lesson.ipynb)) computes every number above with NumPy, checks them against Keras' own functions (`keras.activations.softmax`, `keras.losses.BinaryCrossentropy`…), and plots the four activations and their slopes.

```python
def softmax(z):
    e = np.exp(z - z.max())          # subtracting the max avoids overflow, same result
    return e / e.sum()
```

## 7. On real data

The notebook trains the same three-layer network on 600 points shaped like two interlocking moons, changing only the hidden activation:

| Hidden activation | Training accuracy |
|---|---|
| none (linear) | 87.5% |
| sigmoid | 95.8% |
| tanh | 96.8% |
| ReLU | 97.2% |

Logistic regression scores **87.5%** on the same points, exactly the linear network's score, because the linear network *is* one straight line. Any bend lets the network follow the moons; ReLU does it best and fastest here.

## 8. Where it breaks

- **Wrong output for the loss.** A sigmoid output with MSE trains, slowly and badly. A linear output with binary cross-entropy produces "probabilities" outside 0–1. Use the table in section 4.
- **Sigmoid in deep hidden layers.** Its slope is at most 0.25 and nearly zero for large $|z|$, so gradients shrink layer after layer (the *vanishing gradient* problem). ReLU avoids most of this.
- **Dead ReLUs.** A neuron whose $z$ is negative for every example outputs 0 forever and never learns again. A very large learning rate (08.6) is the usual cause.
- **Softmax for multi-label.** Softmax forces probabilities to compete. If a photo can be both "red" and "mug", use one sigmoid per label.
- **Loss is not the business metric.** Training minimises log loss; ShopWave cares about AUC and voucher cost (lesson 03.7). Watch both.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: softmax of [1, 1, 1] and of [10, 0, 0]. What do they tell you about softmax?
2. By hand: binary cross-entropy for a customer who did **not** churn ($y = 0$) when the model predicted 0.9. Compare with the 2.303 above.
3. Show numerically that adding a fourth linear layer to the moons network still leaves it at logistic-regression accuracy.
4. Choose the output layer and loss for: (a) predicting order value in pounds, (b) predicting which of 5 delivery couriers a parcel used, (c) tagging a photo with any of "gift", "Christmas", "kitchen".

**Quiz** (answers at the end of the notebook)

1. Why does a network with no activation functions collapse to a single linear model?
2. Which activation is the usual default for hidden layers, and why?
3. What does softmax guarantee about its outputs?
4. Which loss for churn? Which for "product type out of 10"?
5. Why does binary cross-entropy punish a confident wrong answer so heavily?

## 10. Recap

- Without activation functions, any stack of layers equals one linear layer: $(xW_1)W_2 = x(W_1W_2)$. On the moons, 87.5%, the same as logistic regression.
- ReLU in hidden layers; sigmoid for a binary output; softmax for one-of-many; linear for regression.
- By hand: softmax[2, 1, 0.1] = [0.659, 0.242, 0.099]; log loss 0.105 for a confident right answer, 2.303 for a confident wrong one.
- Match the loss to the output: MSE ↔ linear, binary cross-entropy ↔ sigmoid, categorical cross-entropy ↔ softmax.

**Next:** [08.4 Backpropagation by hand](../04-backpropagation-by-hand/): how the network uses the loss to fix every weight.
