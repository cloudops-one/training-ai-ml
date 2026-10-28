# 08.11 · The same model in PyTorch (optional)

⏱ about 1.5 hours · Prerequisites: [08.8](../08-cnns-fashion-mnist/) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave hires a machine-learning engineer from a research lab. On day one they ask: *"Is this Keras? We use PyTorch."* A few weeks later you find a pretrained model you want on Hugging Face (Module 09), and its example code is PyTorch too.

You do not need to switch frameworks. But you do need to *read* PyTorch, and to see that it is the same deep learning with different spelling. This optional lesson rebuilds the 08.8 CNN in PyTorch, line by line, and shows the parts that Keras' `model.fit` was doing quietly for you.

## 2. The idea

Keras is an automatic car: you say where to go (`compile`, `fit`) and it changes gear for you. PyTorch is a manual car: you write the training loop yourself, every gear change. That is more typing, and more control: researchers like it because they can change any step, and it is easy to debug with ordinary Python.

The engine is identical: layers with weights, a forward pass, a loss, backpropagation and gradient descent (08.2–08.6).

## 3. Definition

- **Tensor**: PyTorch's array, like a NumPy array that can live on a GPU and remembers how it was computed so it can backpropagate.
- **`Dataset`**: a class that returns one `(input, label)` pair for an index; you write `__len__` and `__getitem__`.
- **`DataLoader`**: groups a `Dataset` into shuffled batches.
- **`nn.Module`**: a model is a class; layers are created in `__init__`, and `forward` says how data flows through them.
- **Device**: where tensors and the model live: `"cuda"` (NVIDIA GPU), `"mps"` (Apple Silicon GPU) or `"cpu"`. Model and data must be on the same device.
- **Logits**: raw output scores before softmax. `nn.CrossEntropyLoss` takes logits and applies softmax itself.
- **State dict**: a dictionary of all learned weights; what you save to disk.
- **Channels-first**: PyTorch image batches are `(batch, channels, height, width)`; Keras uses `(batch, height, width, channels)`.

## 4. Formula

The four steps of one training batch, which `model.fit` hides:

1. `optimizer.zero_grad()`: clear the gradients from the previous batch.
2. `loss = loss_fn(model(x), y)`: forward pass and loss.
3. `loss.backward()`: backpropagation fills in $\partial \text{loss} / \partial w$ for every weight.
4. `optimizer.step()`: update every weight, $w \leftarrow w - \eta \, \partial \text{loss} / \partial w$ (for plain gradient descent; Adam adds momentum and scaling).

Parameter counts follow the same formulas as Keras: $k \cdot k \cdot C_{in} \cdot C_{out} + C_{out}$ for `Conv2d`, $n_{in} \cdot n_{out} + n_{out}$ for `Linear`.

## 5. By hand

**Translating layer by layer:**

| Keras (08.8) | PyTorch | parameters |
|---|---|---|
| `Conv2D(16, 3)` | `nn.Conv2d(1, 16, kernel_size=3)` | 3·3·1·16 + 16 = 160 |
| `MaxPooling2D(2)` | `nn.MaxPool2d(2)` | 0 |
| `Conv2D(32, 3)` | `nn.Conv2d(16, 32, kernel_size=3)` | 3·3·16·32 + 32 = 4,640 |
| `MaxPooling2D(2)` | `nn.MaxPool2d(2)` | 0 |
| `Flatten()` | `nn.Flatten()` | 0 |
| `Dense(64)` | `nn.Linear(800, 64)` | 800·64 + 64 = 51,264 |
| `Dense(10, softmax)` | `nn.Linear(64, 10)` (softmax inside the loss) | 64·10 + 10 = 650 |
| | **total** | **56,714** |

Two differences you must handle by hand:

- PyTorch layers need the **input size** (`Conv2d(16, 32, …)`, `Linear(800, 64)`); Keras works it out. You compute 800 = 32 channels × 5 × 5 yourself, exactly as in 08.8.
- Photos must be **channels-first**: a batch of 128 Fashion-MNIST photos is `(128, 1, 28, 28)`, not `(128, 28, 28, 1)`.

**Keras and PyTorch side by side:**

| task | Keras | PyTorch |
|---|---|---|
| define a model | `keras.Sequential([...])` | `class SmallCNN(nn.Module)` with `forward` |
| feed data | NumPy arrays to `fit` | `Dataset` + `DataLoader` |
| train | `model.fit(x, y, epochs=3)` | your own loop: zero_grad, forward, backward, step |
| choose GPU | automatic | `model.to(device)`, `x.to(device)` |
| prediction mode | automatic in `predict` | `model.eval()` and `torch.no_grad()` |
| save | `model.save("m.keras")` | `torch.save(model.state_dict(), "m.pt")` |

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The model:

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.features(x))
```

And the heart of the training loop:

```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()
```

The notebook picks the device automatically: `cuda` on a PC with an NVIDIA card, `mps` on an Apple Silicon Mac, otherwise `cpu`. The first run downloads its own copy of Fashion-MNIST (about 30 MB) into `data/raw/`.

## 7. On real data

The PyTorch CNN has **56,714** parameters, the same as the Keras one. After 3 epochs it reaches **0.867** test accuracy (0.822, 0.852, 0.867 per epoch). The Keras version in 08.8 trained for 8 epochs and reached 0.896; give this one the same budget and it lands in the same place. Training took a few seconds on the Apple GPU that produced the notebook; on a laptop CPU allow one to three minutes.

The ankle-boot test photo gets 86.1% "ankle boot", 10.6% "sneaker", 2.9% "sandal". In the exercises the dropout version scores 0.856 after 3 epochs, and its most common mistakes are, once more, shirts, coats and pullovers. Changing the framework does not change what is hard about the data.

**Which do companies use?** Both. PyTorch dominates research and most open-source models (Hugging Face models usually ship PyTorch first). Keras is common for teaching, quick prototypes and teams already on TensorFlow, and Keras 3 can itself run on a PyTorch backend. Knowing one well and reading the other is enough.

## 8. Where it breaks

- **Forgetting `zero_grad()`.** PyTorch adds new gradients to old ones. Without clearing them, every update uses the sum of all previous batches.
- **Forgetting `model.eval()`.** Dropout and batch normalisation stay in training mode, so predictions are noisy and worse.
- **Device mismatch.** "Expected all tensors to be on the same device" means the model is on the GPU and the batch is not (or the other way round).
- **Softmax twice.** `nn.CrossEntropyLoss` applies softmax internally. Adding `nn.Softmax` at the end of the model trains badly.
- **Channels last.** Feeding `(N, 28, 28, 1)` to `Conv2d` fails or, worse, treats rows as channels.
- **Different numbers on different devices.** GPU and CPU arithmetic differs slightly, so results vary a little even with `torch.manual_seed`.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Count the parameters of `SmallCNN` by hand and check with `named_parameters()`.
2. Add `nn.Dropout(0.3)` after `Flatten`. Why must you call `model.eval()` before measuring accuracy?
3. Build a confusion matrix for the PyTorch model and list its five most common mistakes.
4. Save a model's state dict, load it into a fresh model, and check the predictions are identical.

**Quiz**

1. What is the shape of a batch of 128 greyscale 28 × 28 photos in Keras, and in PyTorch?
2. Name the four steps inside the PyTorch training loop.
3. What goes wrong if you forget `optimizer.zero_grad()`?
4. What does `model.eval()` do?
5. Why does the PyTorch model end in `nn.Linear(64, 10)` with no softmax?

## 10. Recap

- PyTorch and Keras build the same networks; the CNN has 56,714 parameters in both.
- In PyTorch you write the model as a class with `forward`, feed data through a `Dataset` and `DataLoader`, and write the training loop: zero_grad, forward and loss, backward, step.
- Images are channels-first, and you move model and data to a device (`cuda`, `mps`, `cpu`) yourself.
- `nn.CrossEntropyLoss` takes raw logits; call `model.eval()` before predicting.
- Research and most open models use PyTorch; reading both frameworks is the practical goal.

**Next:** the [module project](../project/): a ShopWave product image classifier, end to end.
