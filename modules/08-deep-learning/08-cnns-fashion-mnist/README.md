# 08.8 · CNNs on Fashion-MNIST

⏱ about 2 hours · Prerequisites: [08.6](../06-training-well/), [08.7](../07-images-and-convolution/) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Back to ShopWave's clothing range. The tagging team has labelled 60,000 past product photos into ten categories, and 10,000 more are waiting. The head of e-commerce asks: *"Can a model tag new photos well enough that the team only checks the doubtful ones?"*

You could train the dense network from 08.5 on flattened photos, and it would do reasonably well. But in 08.7 you saw that a convolution reuses a handful of weights at every position and understands neighbourhoods of pixels. This lesson tests whether that matters in practice: same data, same training budget, dense network against a small CNN.

## 2. The idea

A CNN works like an assembly line of inspectors. The first row of inspectors each looks for one tiny pattern (a vertical edge, a curve) everywhere in the photo. Pooling summarises their reports into a smaller sheet. The next row looks at *those* sheets for combinations: a curve above a straight edge, two parallel lines. After a few rows, a final manager reads the summary and says "ankle boot, 99%".

Nobody tells the inspectors what to look for. Training starts them with random filters and nudges every filter, through backpropagation (08.4), towards whatever helps the final answer.

## 3. Definition

- **Fashion-MNIST**: 70,000 greyscale photos, 28 × 28, of ten product types (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), made by Zalando Research. 60,000 for training, 10,000 for testing, 6,000 per class.
- **Convolutional neural network (CNN)**: a network whose early layers are convolutions (usually followed by pooling) and whose last layers are dense.
- **`Conv2D(filters, kernel_size)`**: a layer of learned filters; each produces one feature map.
- **`MaxPooling2D(2)`**: halves height and width (08.7).
- **`Flatten`**: unrolls the final stack of feature maps into one long vector for the dense layers.
- **Softmax output**: ten probabilities that sum to 1 (08.3); the prediction is the largest.
- **Confusion matrix**: a 10 × 10 table of true class (rows) against predicted class (columns), as in 03.8.

## 4. Formula

Parameters of a convolution layer with $F$ filters of size $k \times k$ on $C$ input channels:

$$k \cdot k \cdot C \cdot F + F$$

Parameters of a dense layer with $n_{in}$ inputs and $n_{out}$ neurons:

$$n_{in} \cdot n_{out} + n_{out}$$

Output side after a convolution without padding, then after 2 × 2 pooling (08.7):

$$n \;\to\; n - k + 1 \;\to\; \left\lfloor \tfrac{n - k + 1}{2} \right\rfloor$$

## 5. By hand

**The dense network:** Flatten → Dense(128) → Dense(10).

| layer | output | parameters |
|---|---|---|
| Flatten | 28·28 = 784 | 0 |
| Dense(128) | 128 | 784·128 + 128 = 100,480 |
| Dense(10) | 10 | 128·10 + 10 = 1,290 |
| **total** | | **101,770** |

**The CNN:** Conv2D(16, 3) → MaxPool → Conv2D(32, 3) → MaxPool → Flatten → Dense(64) → Dense(10).

| layer | output | parameters |
|---|---|---|
| Conv2D(16, 3) | 26 × 26 × 16 | 3·3·1·16 + 16 = 160 |
| MaxPooling2D | 13 × 13 × 16 | 0 |
| Conv2D(32, 3) | 11 × 11 × 32 | 3·3·16·32 + 32 = 4,640 |
| MaxPooling2D | 5 × 5 × 32 | 0 (11 / 2 rounds down to 5) |
| Flatten | 800 | 0 |
| Dense(64) | 64 | 800·64 + 64 = 51,264 |
| Dense(10) | 10 | 64·10 + 10 = 650 |
| **total** | | **56,714** |

The CNN has **56% as many parameters** as the dense network. Note where they live: the two convolution layers together have only 4,800; almost all the rest are in the first dense layer after `Flatten`.

**Baseline:** ten balanced classes, so always guessing one class scores **10%**.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The CNN is eight lines:

```python
cnn = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x_train[..., None], y_train, epochs=8, batch_size=128, validation_split=0.1)
```

`x_train[..., None]` adds the channel axis: (60000, 28, 28) becomes (60000, 28, 28, 1). `model.summary()` prints the same parameter counts as your table.

Both networks train for 8 epochs. On the machine that produced the notebook that took about 3 s and 12 s; on a laptop CPU expect one to three minutes for the CNN. Your accuracies will differ in the third decimal: training on different hardware is not bit-for-bit identical, even with a fixed seed.

## 7. On real data

| | parameters | train acc | validation acc | test acc |
|---|---|---|---|---|
| dense | 101,770 | 0.898 | 0.878 | **0.873** |
| CNN | 56,714 | 0.906 | 0.900 | **0.896** |

The CNN is more accurate with fewer weights, and its validation accuracy stays closer to its training accuracy (gap 0.006 against 0.020 for the dense network): it memorises less.

The confusion matrix says where it fails. The five most common mistakes all involve upper-body clothes:

| true → predicted | count (of 1,000 per class) |
|---|---|
| Shirt → T-shirt/top | 154 |
| Coat → Shirt | 86 |
| Coat → Pullover | 75 |
| T-shirt/top → Shirt | 68 |
| Pullover → Shirt | 65 |

Shirt is the worst class (68.6% correct), then coat (79.4%). Shoes, bags and trousers are almost always right. Looking at the misclassified photos, many mistakes are understandable: at 28 × 28 pixels a shirt and a T-shirt share the same outline.

The last cell shows the 16 learned first-layer filters. They began as random noise; after training several look like the edge detectors you built by hand in 08.7.

## 8. Where it breaks

- **Low resolution limits everyone.** No architecture can recover detail that 28 × 28 pixels do not contain. Real ShopWave photos are larger and in colour (08.9).
- **Accuracy hides the weak classes.** 89.6% overall, but shirts are right only 69% of the time. Report per-class accuracy to the business.
- **Confident mistakes exist.** In the exercises, 141 of the wrong predictions came with over 90% confidence. A confidence threshold reduces errors; it does not remove them.
- **Same distribution assumed.** Fashion-MNIST photos are centred, cropped and on a black background. Supplier photos with models, shadows or busy backgrounds would score far lower until the model sees examples like them.
- **More epochs is not free.** Training longer raises training accuracy faster than validation accuracy (overfitting, 08.6); exercise 2 shows dropout closing that gap.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Count the parameters of a bigger CNN (32 and 64 filters, Dense(128)) by hand, layer by layer.
2. Add dropout and train 15 epochs with and without it. What happens to the train–validation gap?
3. Replace `Flatten` with `GlobalAveragePooling2D`. How many parameters, and what accuracy?
4. How confident is the CNN when it is right, and when it is wrong? What is its top-2 accuracy?

**Quiz**

1. What accuracy does always guessing one class get on Fashion-MNIST?
2. How many parameters does `Conv2D(32, 3)` have when its input has 16 channels?
3. Give two reasons the CNN beats the dense network with fewer parameters.
4. Which product class is hardest, and why?
5. Training accuracy 0.93, validation accuracy 0.91 and the gap is growing. What is happening?

## 10. Recap

- A CNN stacks convolution and pooling layers to find local patterns, then a dense head to classify.
- Count parameters with $k \cdot k \cdot C \cdot F + F$ for convolutions and $n_{in} \cdot n_{out} + n_{out}$ for dense layers; most weights usually sit in the first dense layer.
- On Fashion-MNIST the small CNN reaches 89.6% test accuracy with 56,714 parameters, against 87.3% for a dense network with 101,770.
- The confusion matrix shows the business story: shoes, bags and trousers are reliable; shirts, coats and pullovers get mixed up.
- First-layer filters learned from data look like the edge detectors built by hand in 08.7.

**Next:** [08.9 Data augmentation and transfer learning](../09-augmentation-transfer-learning/) moves to real colour photos with only a few hundred examples per class.
