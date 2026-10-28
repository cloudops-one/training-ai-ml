# 08.9 · Data augmentation and transfer learning

⏱ about 2.5 hours · Prerequisites: [08.8](../08-cnns-fashion-mnist/), bias and variance ([04.9](../../04-trustworthy-models/09-bias-variance/)) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave is adding a "garden & flowers" range. Suppliers upload colour photos of five kinds of flower (daisy, dandelion, rose, sunflower, tulip), and the team has labelled only **200 photos per kind**. The CNN from 08.8 learned from 6,000 photos per class, on tiny greyscale images. Will it cope with 1,000 colour photos in total?

It will not, and this lesson shows it. Then it fixes the problem two ways: by stretching the photos you have (**augmentation**) and by borrowing a network that has already seen millions of photos (**transfer learning**). Transfer learning is how almost every real image project at a company starts.

## 2. The idea

**Augmentation.** A rose flipped left to right is still a rose. So is a rose turned slightly or seen a little closer. Showing the model these variations teaches it that they do not matter, without anyone labelling new photos.

**Transfer learning.** A new employee who already knows photography learns to sort flower photos in an afternoon; someone who has never seen a photograph would need months. A network trained on 1.3 million everyday photos (ImageNet) has already learned edges, textures, petals and leaves in its early and middle layers. You keep those layers, remove its final "1,000 everyday objects" layer, and train a new small layer for your five classes.

## 3. Definition

- **Data augmentation**: random changes (flips, rotations, zooms, crops, colour shifts) applied to training images each time they are used. Keras layers: `RandomFlip`, `RandomRotation`, `RandomZoom`. They are active only during training.
- **Pretrained model**: a network already trained on a large dataset. Keras includes many in `keras.applications`.
- **Base (backbone)**: the pretrained network without its final classification layer. With `pooling="avg"` it turns each photo into a fixed-length vector of **features** (an **embedding**).
- **Head**: the new layer(s) you add on top for your own classes.
- **Frozen**: weights that do not change during training (`base.trainable = False`).
- **Feature extraction**: training only the head on a frozen base.
- **Fine-tuning**: afterwards, unfreezing some top layers of the base and training on with a much smaller learning rate.
- **MobileNetV3Small**: a small pretrained base (939,120 weights, 576 features per photo) designed for phones. **EfficientNetB0**: a larger one (4,049,571 weights, 1,280 features).

## 4. Formula

Trainable weights in a new dense head with $d$ features and $c$ classes:

$$d \cdot c + c$$

Horizontal flip of an image with width $W$: the pixel in column $j$ moves to column $W - 1 - j$.

The learning rate rule for fine-tuning is a rule of thumb, not a formula: use about **1/100 of the original rate** (here $10^{-3} \to 10^{-5}$), so pretrained weights move only a little.

## 5. By hand

**A flip.** On a 3 × 3 array, column $j$ moves to column $2 - j$:

```
1 2 3        3 2 1
4 5 6   →    6 5 4
7 8 9        9 8 7
```

**The head.** MobileNetV3Small gives 576 features per photo; five classes:

$$576 \times 5 + 5 = 2{,}885 \text{ trainable weights}$$

against **939,120** frozen ones in the base. You are training 0.3% of the network.

For EfficientNetB0: $1{,}280 \times 5 + 5 = 6{,}405$.

**The from-scratch CNN** (three Conv2D layers of 16, 32 and 64 filters, global pooling, Dense(5)):

| layer | parameters |
|---|---|
| Conv2D(16, 3) on 3 channels | 3·3·3·16 + 16 = 448 |
| Conv2D(32, 3) | 3·3·16·32 + 32 = 4,640 |
| Conv2D(64, 3) | 3·3·32·64 + 64 = 18,496 |
| Dense(5) on 64 | 64·5 + 5 = 325 |
| **total** | **23,909** |

It has more trainable weights than the transfer head, but it starts from random numbers.

**Memory.** 1,000 photos × 160 × 160 × 3 = 76.8 million numbers. Stored as `uint8` (1 byte each) that is 77 MB; as `float32` it would be 307 MB. The notebook keeps photos as `uint8`, which matters on an 8 GB laptop.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). Augmentation is a small model of its own:

```python
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

Transfer learning is a frozen base, the augmentation, and a new head:

```python
base = keras.applications.MobileNetV3Small(input_shape=(160, 160, 3), include_top=False,
                                           weights="imagenet", pooling="avg")
base.trainable = False
inputs = keras.Input((160, 160, 3))
x = base(augment(inputs), training=False)
outputs = layers.Dense(5, activation="softmax")(layers.Dropout(0.2)(x))
model = keras.Model(inputs, outputs)
```

`include_top=False` drops ImageNet's 1,000-class layer. `training=False` keeps the base's batch-normalisation layers in inference mode, which matters later when you fine-tune. MobileNetV3 rescales pixels itself, so you feed it the raw 0–255 values.

Fine-tuning unfreezes the top 20 layers and recompiles with `learning_rate=1e-5`.

The first run downloads 229 MB of photos and about 10 MB of weights. Each model then trains in roughly 10–40 seconds on the machine that ran the notebook; on a laptop CPU allow a few minutes each. Accuracies move by a point or two between machines.

## 7. On real data

All four models trained on the same 1,000 photos and were tested on the same 500 unseen ones.

| model | trainable weights | test accuracy |
|---|---|---|
| small CNN from scratch | 23,909 | 0.552 |
| small CNN from scratch + augmentation | 23,909 | 0.514 |
| MobileNetV3Small, frozen + augmentation | 2,885 | **0.846** |
| … then fine-tuned (top 20 layers) | 353,429 | 0.848 |
| logistic regression on MobileNetV3Small features | (576 × 5 + 5) | **0.870** |

Three lessons from this table:

1. **Transfer learning is the big win**: from 55% to about 85% by training 2,885 weights instead of 23,909.
2. **Augmentation did not rescue the from-scratch CNN.** Its training accuracy (0.568) is barely above its test accuracy: the model is *underfitting*, not memorising. Augmentation fights overfitting, so it cannot help here. In a 40-epoch check the gap stayed the same (0.65 without augmentation, 0.59 with). A better starting point is the fix, not more tricks.
3. **Fine-tuning added almost nothing** on this data, and plain logistic regression on the frozen features (Module 03) did best. Always try the simple version first.

The confusion matrix shows the remaining mistakes are mostly roses and tulips mixed up (11 and 13 photos): similar colours and shapes.

In the exercises, EfficientNetB0 reaches 0.878 but takes about four times as long, and the data-size curve shows 0.630 accuracy from just 25 photos per class, rising to 0.740 (50), 0.816 (100) and 0.844 (200).

## 8. Where it breaks

- **Domain gap.** ImageNet is everyday photos. A base trained on it transfers well to products and flowers, less well to X-rays, satellite images or microscope slides.
- **Wrong augmentations change the label.** Flipping a T-shirt is fine; flipping a photo of text, or rotating a "6" into a "9", is not. Choose augmentations that keep the meaning.
- **Augmentation is not a cure-all.** It helps an overfitting model. An underfitting one, like the small CNN here, gets worse.
- **Fine-tuning can wreck the base.** A normal learning rate on unfrozen pretrained layers destroys what they learned. Use a tiny rate, unfreeze only the top, and keep batch normalisation in inference mode.
- **Preprocessing must match.** Each pretrained model expects its own input scaling. MobileNetV3 and EfficientNet rescale internally; many others need `keras.applications.<model>.preprocess_input`. A mismatch silently costs accuracy.
- **Licences.** Pretrained weights and datasets carry licences; check them before shipping a model.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: how many trainable weights does a 5-class head on EfficientNetB0 have? Check in code.
2. Swap in EfficientNetB0. Compare accuracy and time with MobileNetV3Small. Which would you put in a phone app?
3. Train the frozen MobileNetV3Small on 25, 50, 100 and 200 photos per class and plot the result.
4. Train the frozen model without augmentation. Does it matter?

**Quiz**

1. What is data augmentation, in one sentence?
2. Are augmentation layers active when the model makes predictions?
3. What is transfer learning?
4. How many trainable weights does a 5-class head on MobileNetV3Small (576 features) have?
5. Why use a much smaller learning rate when fine-tuning?

## 10. Recap

- With a few hundred photos per class, a CNN from random weights struggles (55% here).
- Augmentation creates varied training copies and fights overfitting; it does not help a model that is underfitting.
- Transfer learning reuses a network trained on millions of photos: freeze the base, train a small head. Here it reached about 85% by training 2,885 weights.
- Fine-tuning the top layers with a tiny learning rate can add a little; here it did not.
- A frozen base is a feature extractor: logistic regression on its features scored 87%.

**Next:** [08.10 Sequences: RNNs and LSTMs](../10-sequences-rnn-lstm/) moves from pictures to data where order matters.
