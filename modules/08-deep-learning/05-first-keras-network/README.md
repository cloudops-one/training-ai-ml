# 08.5 · Your first network in Keras

⏱ about 2 hours · Prerequisites: 08.2–08.4, 04.5 (scaling), 03.7 (AUC) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You can now compute a neuron, a layer and a full backward pass by hand. Doing that for a real network with hundreds of weights and thousands of customers would take weeks.

**Keras** writes the forward pass from a short description of your layers, runs backpropagation with `GradientTape` (lesson 08.4), applies the updates, tracks the loss and metrics, and saves the result. It is the same maths you did by hand, packaged so that a network takes ten lines.

The first job: put the ShopWave churn table through a real neural network and see whether it beats the logistic regression you have been benchmarking against since Module 03 (test AUC **0.741**).

## 2. The idea

Every Keras project has the same five steps, in the same order:

```
1. build      describe the layers                    keras.Sequential([...])
2. compile    choose optimiser, loss, metrics          model.compile(...)
3. fit        run forward + backward + update loops    model.fit(X, y, epochs=...)
4. evaluate   score on data it has never seen          model.evaluate(X_test, y_test)
5. predict    use it                                   model.predict(X_new)
```

It is the scikit-learn rhythm (`fit`, `predict`) with one extra step in front: **compile** decides *how* to learn (lessons 08.3 and 08.6), because a network, unlike `LogisticRegression`, has many choices there.

## 3. Definition

- **`keras.Sequential`**: a model whose layers run one after another, each feeding the next.
- **`keras.Input(shape)`**: declares the shape of one example, here `(5,)` for five features.
- **`layers.Dense(units, activation)`**: a fully connected layer (lesson 08.2).
- **Optimizer**: the update rule. `"adam"` is gradient descent with automatic, per-weight step sizes (lesson 08.6).
- **Loss**: what training minimises (lesson 08.3). Binary cross-entropy for churn.
- **Metrics**: numbers Keras reports but does not train on, such as accuracy and AUC.
- **Epoch**: one pass through all the training rows.
- **Batch**: the rows used for one update step. `batch_size=32` means one backward pass and update every 32 customers.
- **`validation_split`**: Keras holds back that fraction of the training rows and reports the loss and metrics on them after every epoch.
- **History**: the per-epoch record that `fit` returns.

## 4. Formula

Counting parameters. A `Dense` layer from $n$ inputs to $k$ neurons has

$$n \times k \ \text{weights} \; + \; k \ \text{biases}.$$

How many updates training does:

$$\text{steps per epoch} = \left\lceil \frac{\text{training rows}}{\text{batch size}} \right\rceil, \qquad \text{total steps} = \text{epochs} \times \text{steps per epoch}$$

## 5. By hand

The churn network: 5 features → 16 ReLU neurons → 8 ReLU neurons → 1 sigmoid output.

| Layer | Calculation | Parameters |
|---|---|---|
| `Dense(16)` | $5 \times 16 + 16$ | 96 |
| `Dense(8)` | $16 \times 8 + 8$ | 136 |
| `Dense(1)` | $8 \times 1 + 1$ | 9 |
| **Total** | | **241** |

Compare logistic regression: 6 parameters (five weights and an intercept). The network has forty times more numbers to learn from the same 2,487 customers.

Training steps: `validation_split=0.2` keeps 80% of the 2,487 training customers for learning, which is 1,989 rows. With `batch_size=32` that is $\lceil 1989 / 32 \rceil = 63$ steps per epoch, and 50 epochs make **3,150** forward-backward-update rounds, each the calculation you did by hand in lesson 08.4, for 241 weights at once.

## 6. In code

```python
model = keras.Sequential([
    keras.Input((5,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
history = model.fit(Xs_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=0)
model.evaluate(Xs_test, y_test)            # loss, accuracy, AUC on unseen customers
model.predict(Xs_test)                     # churn probabilities, one per customer
```

`Xs_train` and `Xs_test` are the features scaled with a `StandardScaler` fitted **on the training customers only** (lesson 04.5). Neural networks need scaled inputs even more than logistic regression does: unscaled spend in pounds would dominate every weighted sum.

The notebook ([lesson.ipynb](lesson.ipynb)) also prints `model.summary()`, which shows 96, 136 and 9 parameters, total 241, as computed above, and saves the trained model to the repo's `models/` folder and loads it back.

## 7. On real data

Fifty epochs take about two seconds on a laptop. On the 830 test customers:

| Model | Parameters | Test AUC |
|---|---|---|
| Logistic regression (Module 03) | 6 | **0.741** |
| Neural network 5-16-8-1 | 241 | **0.740** |

A tie. The training curves show why: the validation AUC is already 0.742 after the first epoch and peaks at about 0.754 around epoch 10, then drifts down while the training AUC keeps creeping up (0.757 by epoch 50). The network learns everything this table has to offer almost immediately, and the extra capacity only starts memorising the training customers.

This matches what Module 05 found with boosted trees. Five well-chosen columns with mostly straight-line relationships leave nothing for extra layers to discover. Deep learning earns its keep on images (lessons 08.7–08.9) and text (Module 09), not here, and that is a perfectly good result to report.

## 8. Where it breaks

- **Unscaled inputs.** Train on raw pounds and days and the network learns slowly or not at all. Always scale, and fit the scaler on training rows only.
- **Wrong final layer or loss.** `Dense(1)` without a sigmoid plus binary cross-entropy gives meaningless "probabilities". Use the table in lesson 08.3.
- **Validation split on ordered data.** `validation_split` takes the *last* 20% of rows, without shuffling. If your rows are sorted (by date, by customer value), shuffle first or pass `validation_data=` explicitly.
- **Trusting one run.** Different random seeds give AUCs a few thousandths apart. Before claiming a 0.003 improvement, rerun with other seeds or use cross-validation (lesson 04.7).
- **Bigger is not better on small tables.** Exercise 3 shows a much larger network does no better.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: count the parameters of 5 → 64 → 64 → 1. Check with `model.summary()`.
2. Retrain the churn network with `batch_size=8` and `batch_size=256`. How many update steps per epoch does each make, and how long does each take?
3. Train the 5 → 64 → 64 → 1 network for 50 epochs. Does it beat 0.741?
4. Turn the network's test probabilities into the voucher decision from lesson 03.7: flag customers above a threshold of 0.5 and compute precision and recall.

**Quiz** (answers at the end of the notebook)

1. What are the five steps of every Keras project?
2. What does `compile` decide that scikit-learn models do not ask you about?
3. How many parameters does `Dense(8)` have after a `Dense(16)` layer?
4. What is an epoch? What is a batch?
5. The network tied logistic regression on churn. Give one reason why.

## 10. Recap

- Keras' rhythm: **build → compile → fit → evaluate → predict**.
- A `Dense` layer from $n$ to $k$ has $nk + k$ parameters; the 5-16-8-1 churn network has 241, matching `model.summary()`.
- Scale inputs on the training rows only; pick the output and loss from lesson 08.3.
- On ShopWave churn the network ties logistic regression (test AUC 0.740 vs 0.741) and starts memorising after about 10 epochs. Honest and expected for a small table.
- Save with `model.save("models/name.keras")`, reload with `keras.models.load_model`.

**Next:** [08.6 Training well](../06-training-well/): what to do when a network learns too slowly, too wildly, or too well.
