# 08.7 · Images as numbers; convolution by hand

⏱ about 2 hours · Prerequisites: [08.5](../05-first-keras-network/), [08.6](../06-training-well/) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave is launching a clothing range. Suppliers send thousands of product photos a week, and each one must be tagged with a category ("trousers", "sneaker", "bag") before it can go on the website. Today two people do it by hand. They are slow, they disagree with each other, and they hate it.

You already know how to build a dense network (08.5). So why not flatten each photo into one long row of pixels and feed it in? Because a dense layer treats every pixel as a separate, unrelated input. It has no idea that pixel 300 sits right next to pixel 301, and it must learn "an edge at the top left" and "the same edge at the bottom right" as two different things. Photos need a layer that understands *neighbourhoods*. That layer is the **convolution**, and this lesson computes one by hand.

## 2. The idea

Imagine checking a fabric roll for flaws with a small magnifying glass. You slide the glass across the roll, square by square, and at each position ask the same question: *is there a flaw under the glass?* You use the same glass, and the same question, everywhere.

A convolution does exactly that. A small grid of numbers, the **filter**, slides over the image. At each position it multiplies the pixels under it by its own numbers and adds them up. Big result: "the pattern I look for is here". Small result: "not here". The output is a new image, a **feature map**, showing where the pattern appears.

A CNN (convolutional neural network) stacks these: early filters find edges, later ones combine edges into corners and textures, and later still into "collar", "sole" or "strap".

## 3. Definition

- **Image as an array**: height × width × channels. A greyscale photo has 1 channel; a colour photo has 3 (red, green, blue). Fashion-MNIST photos are 28 × 28 × 1, with pixel values 0–255.
- **Pixel scaling**: dividing by 255 so inputs lie between 0 and 1. The picture is unchanged; training is more stable.
- **Filter (kernel)**: a small grid of weights, usually 3 × 3, with one 3 × 3 slice per input channel.
- **Convolution**: slide the filter over the image; each output cell is the sum of (patch × filter). Deep-learning libraries do not flip the filter first, so strictly this is *cross-correlation*, but everyone calls it convolution.
- **Feature map**: the output grid of one filter.
- **Stride** $s$: how many pixels the filter jumps each step (1 = every position, 2 = every other).
- **Padding** $p$: zeros added around the border so the filter can sit on edge pixels and the output keeps its size.
- **Max pooling**: split the map into small blocks (usually 2 × 2) and keep only the largest value in each. It halves the size and has nothing to learn.

## 4. Formula

One output cell, for a 3 × 3 filter $K$ placed with its top-left corner at row $i$, column $j$:

$$\text{out}[i, j] = \sum_{a=0}^{2}\sum_{b=0}^{2} \text{image}[i + a,\ j + b] \times K[a, b]$$

Output size along one side, for input size $n$, filter size $k$, padding $p$ and stride $s$:

$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

Weights in a convolution layer with $F$ filters of size $k \times k$ on $C$ input channels:

$$k \times k \times C \times F \;+\; F \text{ biases}$$

## 5. By hand

**The image** is a 5 × 5 picture of a bright square (9) on a dark background (0) that runs off the right-hand edge:

```
0 0 0 0 0
0 0 9 9 9
0 0 9 9 9
0 0 9 9 9
0 0 0 0 0
```

**The filter** detects vertical edges where it gets brighter from left to right:

```
-1 0 1
-1 0 1
-1 0 1
```

Multiplying by this filter and summing is the same as *(sum of the right column of the patch) − (sum of the left column)*.

**Step 1: output size.** $n = 5, k = 3, p = 0, s = 1$: $\lfloor (5 - 3)/1 \rfloor + 1 = 3$. The output is 3 × 3.

**Step 2: the top-left cell.** The patch is image rows 0–2, columns 0–2:

```
0 0 0        -1 0 1        -0 0 0
0 0 9   ×    -1 0 1   =    -0 0 9      sum = 18
0 0 9        -1 0 1        -0 0 9
```

**Step 3: every cell.** Column sums of each 3-row band make this quick:

| band | col 0 | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|---|
| rows 0–2 | 0 | 0 | 18 | 18 | 18 |
| rows 1–3 | 0 | 0 | 27 | 27 | 27 |
| rows 2–4 | 0 | 0 | 18 | 18 | 18 |

Each output = (column $j+2$) − (column $j$):

| | out col 0 | out col 1 | out col 2 |
|---|---|---|---|
| out row 0 | 18 − 0 = **18** | 18 − 0 = **18** | 18 − 18 = **0** |
| out row 1 | **27** | **27** | **0** |
| out row 2 | **18** | **18** | **0** |

The filter fires (18–27) where the dark background meets the square's left side, and gives 0 inside the square, where nothing changes. There is no right-hand edge in the picture (the square runs off the image), so nothing fires there.

**Step 4: stride and padding.** With stride 2: $\lfloor (5 - 3)/2 \rfloor + 1 = 2$, so only the corner windows survive: `[[18, 0], [18, 0]]`. With padding 1: $\lfloor (5 + 2 - 3)/1 \rfloor + 1 = 5$, so the output stays 5 × 5. The added zeros create a new "edge" at the right border, which shows up as negative numbers there.

**Step 5: max pooling.** Take this 4 × 4 map and keep the largest number in each 2 × 2 block:

```
1 3 | 2 0
4 6 | 1 1          6  2
----+----    →
0 2 | 5 7          2  8
1 1 | 3 8
```

**Step 6: weights.** A `Conv2D(16, 3)` layer on a greyscale image has 3 × 3 × 1 × 16 + 16 = **160** weights, however large the photo is. A `Dense(16)` layer on a flattened 28 × 28 photo needs 784 × 16 + 16 = 12,560.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). It:

1. loads a Fashion-MNIST photo and prints its numbers;
2. builds a tiny colour image to show the channel axis;
3. computes the 5 × 5 convolution with a ten-line NumPy function, then with a Keras `Conv2D` whose weights are set by hand (they match);
4. checks the output-size formula against Keras for six cases;
5. does max pooling in one line of NumPy and in Keras.

The Keras call is the important one:

```python
conv = layers.Conv2D(filters=1, kernel_size=3, use_bias=False)
conv.build((None, 5, 5, 1))                       # batch, height, width, channels
conv.set_weights([vertical_edge.reshape(3, 3, 1, 1)])
conv(image.reshape(1, 5, 5, 1))                   # → the same 3 × 3 output
```

In 08.8 you stop setting the weights: the network learns them.

## 7. On real data

The notebook applies four hand-made filters (vertical edges, horizontal edges, blur, sharpen) to an ankle-boot photo. Each output is 26 × 26: a 3 × 3 filter without padding loses one pixel on every side. The edge maps light up along the boot's outline and stitching and stay at 0 on flat areas.

Two facts about the data from the notebook: the average pixel is 72.9 out of 255, and 50.2% of all pixels are exactly 0 (the black background).

The last cell pushes a blank image through the CNN you will train in 08.8 and prints the shapes: 28 → 26 → 13 → 11 → 5, ending with 5 × 5 × 32 = 800 numbers for the final classifier.

## 8. Where it breaks

- **Borders.** Without padding, edge pixels are seen by fewer filter positions and the image shrinks every layer. With zero padding, the border itself can look like an edge (you saw −9 to −27 appear at the right).
- **Pooling rounds down.** An 11 × 11 map pooled 2 × 2 becomes 5 × 5; the last row and column are dropped.
- **Hand-made filters are not enough.** Edge and blur filters are fixed. Telling a shirt from a T-shirt needs hundreds of filters tuned to this exact data, which is why we learn them.
- **Position is only partly ignored.** Sliding the same filter everywhere makes CNNs good at "a sole somewhere in the picture". A CNN trained only on centred products can still fail on off-centre ones; 08.9 deals with that.
- **Colour photos cost 3×.** Every first-layer filter has a slice per channel, so 3 × 3 becomes 3 × 3 × 3 = 27 weights.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: apply the horizontal-edge filter (the same filter turned on its side) to the 5 × 5 image. What does ReLU do to the result?
2. Compute three output sizes with the formula, including the first layer of ResNet (224 × 224, 7 × 7 filter, padding 3, stride 2). Check them in Keras.
3. Run the 5 × 5 convolution with stride 2 in NumPy and in Keras.
4. Count the weights of `Conv2D(64, 3)`, `Conv2D(64, 5)` and `Dense(64)` on a 224 × 224 colour photo.

**Quiz**

1. What is the shape of a 160 × 160 colour photo as an array, and how many numbers does it hold?
2. Why divide pixel values by 255 before training?
3. What size is the output of a 5 × 5 filter on a 32 × 32 image with no padding and stride 1?
4. What does 2 × 2 max pooling do to a feature map, and how many weights does it learn?
5. Why does a convolution layer need far fewer weights than a dense layer on the same photo?

## 10. Recap

- An image is an array: height × width × channels. Scale pixels to 0–1 before training.
- A convolution slides a small filter over the image; each output is the sum of patch × filter. The output (a feature map) shows where the filter's pattern appears.
- $n_{out} = \lfloor (n + 2p - k)/s \rfloor + 1$. Padding keeps the size; stride shrinks it.
- Max pooling keeps the largest value in each block: half the size, no weights.
- A convolution reuses the same few weights at every position, which is why it suits photos.

**Next:** [08.8 CNNs on Fashion-MNIST](../08-cnns-fashion-mnist/) lets the network learn its own filters and compares it with a dense network.
