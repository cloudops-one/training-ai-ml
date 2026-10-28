# Keras (and PyTorch)

Keras 3 API. Learn it in [08.5](../modules/08-deep-learning/05-first-keras-network/) → [08.9](../modules/08-deep-learning/09-augmentation-transfer-learning/).

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # quieter TensorFlow start-up
import keras
from keras import layers

keras.utils.set_random_seed(42)              # repeatable (results still vary slightly by machine)
```

## Two ways to build a model

```python
# Sequential: a straight stack
model = keras.Sequential([
    keras.Input(shape=(5,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

# Functional: branches, several inputs or outputs
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()
```

## Layers

| Layer | Does | Typical use | Parameters |
|---|---|---|---|
| `Dense(units)` | weighted sum of every input + bias | tabular data, the final "head" | $(\text{inputs}+1)\times\text{units}$ |
| `Conv2D(filters, k)` | slides $k\times k$ filters over an image | images | $(k\cdot k\cdot C_{in}+1)\times\text{filters}$ |
| `MaxPooling2D(2)` | keeps the max of each 2×2 patch, halves height and width | images | 0 |
| `GlobalAveragePooling2D()` | averages each feature map to one number | before the head in CNNs; far fewer weights than `Flatten` | 0 |
| `Flatten()` | turns a grid into one long vector | small CNNs | 0 |
| `Dropout(rate)` | randomly zeroes inputs during training | fighting overfitting | 0 |
| `BatchNormalization()` | re-centres and rescales activations per batch | deeper nets, faster training | $4\times\text{channels}$ (2 trainable) |
| `Embedding(vocab, dim)` | looks up a vector per token id | text | $\text{vocab}\times\text{dim}$ |
| `LSTM(units)` | reads a sequence with gated memory | text, time series | $4\,(\text{units}\cdot(\text{inputs}+\text{units})+\text{units})$ |
| `RandomFlip`, `RandomRotation`, `RandomZoom` | augmentation, active only in training | small image datasets | 0 |

Conv output size: $\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$ for input $n$, padding $p$, kernel $k$, stride $s$. → [08.7](../modules/08-deep-learning/07-images-and-convolution/)

Check: `Dense(128)` on 784 pixels then `Dense(10)` = 100,480 + 1,290 = **101,770** parameters, as in [08.8](../modules/08-deep-learning/08-cnns-fashion-mnist/).

## Output layer and loss, by task

| Task | Last layer | Loss | Labels look like |
|---|---|---|---|
| Regression | `Dense(1)` (no activation) | `"mse"` or `"mae"` | numbers |
| Binary classification | `Dense(1, activation="sigmoid")` | `"binary_crossentropy"` | 0 / 1 |
| Multi-class, integer labels | `Dense(K, activation="softmax")` | `"sparse_categorical_crossentropy"` | 0 … K−1 |
| Multi-class, one-hot labels | `Dense(K, activation="softmax")` | `"categorical_crossentropy"` | [0, 1, 0, …] |
| Multi-label (several tags at once) | `Dense(K, activation="sigmoid")` | `"binary_crossentropy"` | [1, 0, 1, …] |

Hidden layers: `relu` by default. Why non-linear activations matter: [08.3](../modules/08-deep-learning/03-activations-and-losses/).

## Compile, fit, evaluate, predict

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("models/best.keras", monitor="val_loss", save_best_only=True),
]
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50,
                    batch_size=32, callbacks=callbacks, verbose=0)

model.evaluate(X_test, y_test, verbose=0)
p = model.predict(X_test, verbose=0).ravel()
model.save("models/churn_net.keras")
model = keras.models.load_model("models/churn_net.keras")
```

Scale tabular inputs first (`StandardScaler` fitted on train). Plot `history.history["loss"]` vs `["val_loss"]`: diverging curves = overfitting. → [08.6](../modules/08-deep-learning/06-training-well/)

## Training problems

| Symptom | Try |
|---|---|
| Loss explodes or is `nan` | Lower learning rate (1e-3 → 1e-4); scale inputs |
| Loss barely moves | Raise learning rate; Adam instead of SGD; more epochs |
| Train ≫ validation | Dropout, L2 (`kernel_regularizer=keras.regularizers.l2(1e-4)`), augmentation, EarlyStopping, more data |
| Train ≈ validation, both poor | Bigger model, more epochs, better features; augmentation will not help (underfitting) |
| Tabular data and still losing to logistic regression | Normal: *0.740 vs 0.741 on churn*. Use the simpler model. |

## Transfer learning

```python
base = keras.applications.MobileNetV3Small(include_top=False, weights="imagenet",
                                           input_shape=(224, 224, 3), pooling="avg")
base.trainable = False                                   # 1. freeze: train the head only

inputs = keras.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = base(x, training=False)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

base.trainable = True                                    # 2. optional fine-tune:
for layer in base.layers[:-20]:                          #    unfreeze the top layers only
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),     #    re-compile, tiny learning rate
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

*1,000 flower photos: from scratch 0.552, frozen MobileNetV3Small 0.846.* MobileNetV3 and EfficientNet include their own input rescaling. → [08.9](../modules/08-deep-learning/09-augmentation-transfer-learning/)

## The same thing in PyTorch

| Keras | PyTorch |
|---|---|
| `layers.Dense(16)` | `nn.Linear(in_features, 16)` (you give the input size) |
| `layers.Conv2D(32, 3)` | `nn.Conv2d(in_channels, 32, 3)`; images are (batch, **channels**, H, W) |
| `activation="relu"` | `nn.ReLU()` or `F.relu(x)` in `forward` |
| `Sequential([...])` / functional model | `nn.Sequential(...)` or a class with `__init__` + `forward` |
| `compile(optimizer, loss)` | `opt = torch.optim.Adam(model.parameters(), 1e-3)`; `loss_fn = nn.CrossEntropyLoss()` (takes raw logits) |
| `fit(...)` | your own loop: `opt.zero_grad(); loss.backward(); opt.step()` |
| `predict(X)` | `model.eval()` + `with torch.no_grad(): model(X)` |
| `model.save(...)` | `torch.save(model.state_dict(), path)` |
| GPU used automatically | `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"` |

Industry: PyTorch dominates research and most new projects; Keras/TensorFlow is common in production and teaching. → [08.11](../modules/08-deep-learning/11-pytorch/)
