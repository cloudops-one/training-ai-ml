# 08.6 · Training well: epochs, batches, learning rate, dropout, early stopping

⏱ about 2 hours · Prerequisites: 08.5, 04.9 (bias and variance), 02.4 (learning rate) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Your first network trained without trouble because the churn table is small and forgiving. Most networks are not so easy. The same code can:

- **learn nothing**: the loss jumps around or sits still;
- **learn painfully slowly**: still improving after hundreds of epochs;
- **learn too well**: near-perfect on training customers, worse than logistic regression on new ones.

Each failure has a recognisable signature in the training curves, and a standard fix. This lesson is the troubleshooting guide you will use for every network after this one, including the image models in lessons 08.8 and 08.9.

## 2. The idea

Training a network is walking downhill in fog (lesson 02.4), with three knobs:

- **Learning rate**: the length of each stride. Too long and you leap over the valley and bounce up the far side. Too short and you shuffle forward for hours.
- **Batch size**: how many customers you look at before each step. Few customers give a quick but shaky sense of "downhill"; many give a steady but slow one.
- **Epochs**: how long you keep walking. Walk too long on the training data and you start following its bumps (its noise) rather than the slope of the land.

And three ways to stop a network memorising:

- **Dropout**: randomly switch off half the neurons at each step, so no neuron can rely on a particular partner. Like a team that trains with random players missing.
- **L2 regularisation**: add a small cost for large weights, as Ridge did in lesson 04.10.
- **Early stopping**: keep watching a validation set and stop when it stops improving, keeping the best weights seen.

## 3. Definition

- **Training curve**: training loss per epoch. **Validation curve**: validation loss per epoch, on customers the weights never learned from.
- **Underfitting**: both curves high and close together. Fix: a bigger network, more epochs, a better learning rate.
- **Overfitting**: training loss keeps falling while validation loss turns upward. Fix: more data, dropout, L2, early stopping, a smaller network.
- **Optimiser**: the rule that turns gradients into updates.
  - **SGD**: step $= -\eta \times$ gradient (lesson 02.4).
  - **Momentum**: keep a running "velocity" so consistent gradients build up speed and noisy ones cancel out.
  - **Adam**: momentum plus an automatic step size for each weight. The safe default.
- **Dropout($p$)**: during training, each output of the layer is set to 0 with probability $p$, and the survivors are scaled by $1/(1-p)$. Switched off at prediction time.
- **L2 (weight decay)**: adds $\lambda \sum w^2$ to the loss.
- **Early stopping**: stop when the validation loss has not improved for `patience` epochs; `restore_best_weights=True` rolls back to the best epoch.
- **Checkpoint**: saving the model to disk whenever the validation score improves.
- **Batch normalisation**: a layer that rescales each batch's activations to mean 0 and standard deviation 1. Mainly speeds training up in deep networks; it is not chiefly a cure for overfitting.

## 4. Formula

**Momentum** (with velocity $u$, momentum $\beta$, usually 0.9):

$$u \leftarrow \beta u - \eta \, g, \qquad w \leftarrow w + u$$

**Dropout** at training time, for an output $a$:

$$a' = \begin{cases} 0 & \text{with probability } p \\ a / (1-p) & \text{otherwise} \end{cases}$$

Scaling the survivors keeps the *expected* value of $a'$ equal to $a$, so nothing needs changing at prediction time.

**L2-regularised loss**:

$$L_{\text{total}} = L_{\text{data}} + \lambda \sum_j w_j^2$$

## 5. By hand

**Momentum versus plain SGD.** One weight, a steady gradient $g = 1$ at every step, $\eta = 0.1$, $\beta = 0.9$, starting at $w = 0$.

| Step | SGD: $w$ | Momentum: $u$ | Momentum: $w$ |
|---|---|---|---|
| 1 | −0.10 | $0.9(0) - 0.1 = -0.100$ | −0.100 |
| 2 | −0.20 | $0.9(-0.1) - 0.1 = -0.190$ | −0.290 |
| 3 | −0.30 | $0.9(-0.19) - 0.1 = -0.271$ | −0.561 |

When the gradient keeps pointing the same way, momentum accelerates: nearly twice as far after three steps. When gradients flip back and forth (noise), the velocity terms cancel and momentum calms the zig-zag.

**Dropout.** A layer outputs $[2.0,\ 0.5,\ 1.0,\ 3.0]$. With $p = 0.5$, suppose the random mask keeps neurons 1 and 3. The survivors are doubled ($1/(1-0.5) = 2$): $[4.0,\ 0,\ 2.0,\ 0]$. On average over many masks, each output is still its original value.

**L2 penalty.** Weights $[0.5,\ -1.2,\ 2.0]$ with $\lambda = 0.001$: $0.001 \times (0.25 + 1.44 + 4.0) = 0.00569$ added to the loss. The largest weight contributes most, so training is pushed to keep weights small.

**Early stopping.** Validation losses by epoch, `patience=2`:

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Validation loss | 0.62 | **0.59** | 0.60 | 0.63 | (not run) |

Epoch 2 is the best. Epochs 3 and 4 do not improve on it: that is 2 epochs of patience used up, so training stops after epoch 4 and, with `restore_best_weights=True`, the model goes back to its epoch-2 weights.

## 6. In code

```python
model = keras.Sequential([
    keras.Input((5,)),
    layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss="binary_crossentropy")

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("models/best.keras", monitor="val_loss", save_best_only=True),
]
model.fit(X, y, epochs=200, validation_data=(X_val, y_val), callbacks=callbacks)
```

## 7. On real data

The notebook ([lesson.ipynb](lesson.ipynb)) runs four experiments on the ShopWave churn table.

**Learning rate** (a 5-16-1 network, Adam, 30 epochs):

| Learning rate | Training loss: epoch 1 → 10 → 30 | Best validation loss |
|---|---|---|
| 1.0 | 2.400 → 0.673 → 0.689 | 0.936 |
| 0.1 | 0.617 → 0.587 → 0.596 | 0.588 |
| 0.01 | 0.611 → 0.575 → 0.569 | 0.589 |
| 0.001 | 0.691 → 0.582 → 0.574 | 0.594 |
| 0.0001 | 0.720 → 0.665 → 0.610 | 0.609 |

1.0 is far too big: the loss explodes in the first epoch and never recovers. 0.0001 is too small: still clearly falling after 30 epochs. The middle values all work; Adam is forgiving.

**Optimiser** (same network, learning rate 0.01, training loss after epochs 1 / 5 / 30): SGD 0.705 / 0.628 / 0.585; SGD with momentum 0.647 / 0.583 / 0.575; Adam 0.611 / 0.578 / 0.569. Momentum and Adam get there faster.

**Overfitting on purpose.** Train a deliberately oversized network (two layers of 256 neurons, 67,585 parameters) on only **300** customers for 200 epochs, validating on the other 2,187 training customers:

| Version | Final training loss | Best validation loss (epoch) | Final validation loss | Test AUC |
|---|---|---|---|---|
| Plain | 0.159 | 0.595 (2) | 2.349 | 0.652 |
| Dropout 0.5 | 0.433 | 0.594 (4) | 1.031 | 0.684 |
| L2 0.001 | 0.389 | 0.640 (15) | 0.999 | 0.684 |
| Batch normalisation | 0.037 | 0.613 (25) | 1.326 | 0.711 |
| Early stopping (patience 10) | stopped after 12 epochs, best epoch 2 | | | **0.730** |

The plain network memorises its 300 customers (training loss 0.159) while its validation loss climbs to four times its best. Dropout and L2 slow the damage; batch normalisation lets it fit the training set even harder (training loss 0.037) and is not the right tool here. **Early stopping** wins clearly: it keeps the weights from epoch 2, before memorising began. The notebook also trains logistic regression on the same 300 customers: test AUC 0.736, slightly better than the best network. With little data, the simplest model is hard to beat.

## 8. Where it breaks

- **Tuning on the test set.** Choosing the learning rate, dropout or the early-stopping epoch by looking at test scores leaks the test set into training (lesson 04.2). Use a validation set; touch the test set once.
- **`validation_split` with ordered rows.** It takes the last rows without shuffling. Shuffle first or pass `validation_data`.
- **Patience too short.** Validation curves are noisy; stopping at the first wobble can stop too early. 5–10 epochs is typical.
- **Dropout in the output layer, or at prediction time.** Never on the output; Keras switches it off automatically for `predict` and `evaluate`.
- **Too many knobs at once.** Change one thing per run and keep notes, or use the tuning tools from lesson 05.6.

**A recipe that works most of the time**

1. Scale inputs; choose the output and loss from lesson 08.3.
2. Start with Adam at its default learning rate (0.001) and a network that is a bit too big.
3. Train with a validation set and `EarlyStopping(patience=10, restore_best_weights=True)`.
4. Plot both curves. Underfitting → bigger network or more epochs. Overfitting → dropout, L2, more data.
5. Only then try other learning rates and batch sizes, one at a time.
6. Report the test score once, at the end.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: continue the momentum table for steps 4 and 5. How far has momentum moved compared with SGD?
2. Try learning rates 3.0 and 0.03 on the 5-16-1 churn network. Which row of the table does each resemble?
3. On the oversized 300-customer network, combine dropout 0.5 *and* early stopping. Better than either alone?
4. Use `ModelCheckpoint` to save the best epoch of the 300-customer network to `models/`, then load it and compute its test AUC.

**Quiz** (answers at the end of the notebook)

1. Training loss falls, validation loss rises. What is happening, and name two fixes.
2. What happens with a learning rate that is far too large? Far too small?
3. Why does dropout scale the surviving outputs by $1/(1-p)$?
4. What does `restore_best_weights=True` do?
5. Why must early stopping watch a validation set rather than the test set?

## 10. Recap

- Read the curves: both high = underfitting; training down while validation up = overfitting.
- Learning rate: 1.0 exploded, 0.0001 crawled, 0.001–0.1 all worked with Adam. Momentum and Adam beat plain SGD.
- Dropout, L2 and early stopping fight overfitting. On 300 customers early stopping did best: test AUC 0.730 against 0.652 for the plain network.
- Batch normalisation speeds up deep networks; it is not the main cure for overfitting.
- Always tune on a validation set, report on the test set once.

**Next:** [08.7 Images as numbers; convolution by hand](../07-images-and-convolution/): the layer that made deep learning work on photos.
