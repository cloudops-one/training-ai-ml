# 08.1 · When deep learning wins, and when it does not

⏱ about 30 minutes · Prerequisites: Modules 02–05 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's catalogue team tags every new product photo by hand: *mug*, *cushion*, *lantern*, *bag*, *card*. At 4,000 new products a month that is a full-time job, and the tags are inconsistent ("candle holder" or "lantern"?).

You have a toolbox full of models now: logistic regression, trees, forests, boosting. They all need a **table**. So the obvious first attempt is to turn each photo into a row of hand-made columns: how bright it is, how much of it is white, whether it is wider than tall, whether the left half mirrors the right. Then train a model on those columns.

That attempt fails, and understanding *why* it fails is the whole reason deep learning exists.

## 2. The idea

Think about how you tell a mug from a vase. You do not measure average brightness. You notice a **handle**, a **rim**, a **cylinder**. Those are not pixels; they are shapes made of edges, which are made of pixels.

A deep network learns exactly that ladder, from data:

```
pixels → edges and colours → corners, curves, textures → handles, rims, labels → "mug"
 layer 0        layer 1                 layer 2                  layer 3        output
```

Each layer builds slightly more meaningful features out of the layer below. Nobody designs them. That is what **deep** means: many layers of learned features stacked on top of each other.

On a churn table the columns are already meaningful (recency, frequency, spend), so there is no ladder to climb. That is why neural networks do not beat logistic regression or boosted trees there.

## 3. Definition

- **Structured (tabular) data**: rows and columns where each column already means something: `days_since_last`, `country`, `spend`. Most business data.
- **Unstructured data**: images, audio, free text, video. The raw units (pixels, sound samples, characters) mean almost nothing alone; meaning lives in patterns across many of them.
- **Feature engineering**: a person designing input columns (Module 04.6).
- **Representation learning**: the model learning its own features from raw input. Deep learning is representation learning with many layers.
- **Neural network**: layers of simple units (neurons), each computing a weighted sum and a squash. Lesson 08.2 builds one.
- **Deep**: more than one hidden layer between input and output. Modern image networks have dozens to hundreds.

## 4. Formula

No new maths yet. One number to keep in mind: the **size of the input**.

| Input | Numbers per example |
|---|---|
| ShopWave churn row | 5 |
| 8×8 greyscale digit | 64 |
| 28×28 Fashion-MNIST photo | 784 |
| 224×224 colour product photo | 224 × 224 × 3 = **150,528** |

A model must find the useful patterns among those numbers. With 5 meaningful columns, a person can do the feature work. With 150,528 meaningless ones, only a learner can.

## 5. By hand

Below is an 8×8 handwritten digit, the first one in scikit-learn's digits set. Each number is ink darkness from 0 (white) to 16 (black).

```
 0  0  5 13  9  1  0  0
 0  0 13 15 10 15  5  0
 0  3 15  2  0 11  8  0
 0  4 12  0  0  8  8  0
 0  5  8  0  0  9  8  0
 0  4 11  0  1 12  7  0
 0  2 14  5 10 12  0  0
 0  0  6 13 10  0  0  0
```

Squint and you can see it: a **0**. Now try to write rules a computer could use for all ten digits:

- *"A zero has ink around the edge and none in the middle."* An 8 also has a hole; a 6 has one at the bottom.
- *"A one has ink only in the middle columns."* Unless it is written slanted.
- *"Symmetry left to right."* 0, 8 and sometimes 3 qualify.

Each rule is half true. Now imagine the same exercise for *mug vs vase vs lantern* in colour photos taken from any angle. This is the wall the notebook shows you with real numbers.

## 6. In code

The notebook ([lesson.ipynb](lesson.ipynb)) runs one experiment on 1,797 handwritten digits:

1. **Six hand-made features** per image (average ink, top-minus-bottom ink, left-right symmetry, inked pixel count, left-minus-right ink, ink in the centre) fed to logistic regression: **68%** accuracy.
2. **All 64 raw pixels** fed to the same logistic regression: **97.8%**. The model learns its own weight for every pixel, which beats anything designed by hand.
3. **A small neural network** (one hidden layer of 64 units) on the raw pixels: **97.3%**, trained in under a second.

Two lessons. Letting the model learn from raw input beats hand-made features by thirty points. And on tiny, tidy 8×8 images, even a linear model learning from pixels is enough; the neural network does not win *here*. Deep networks pull ahead as images become larger, messier and more varied, which you will see on clothing photos in lesson 08.8 and product photos in 08.9.

## 7. On real data

The notebook finishes with an honest scoreboard from this course so far:

| Problem | Data | Best simple model | Neural network |
|---|---|---|---|
| Churn (Module 05, 08.5) | 5-column table | Logistic regression, AUC 0.741 | AUC 0.740 |
| Digits (this lesson) | 64 pixels | Logistic on pixels, 97.8% | 97.3% |
| Clothing photos (08.8) | 784 pixels | | CNN, well ahead |
| Product photos (08.9, 08.P) | 150,528 numbers | | transfer learning |

## 8. Where it breaks

Deep learning is the wrong choice when:

- **The data is a small table.** Gradient boosting or logistic regression will match it, train in seconds, and be easier to explain (Module 05.8).
- **You have little labelled data and no pretrained model.** Networks have thousands to millions of weights to learn. A few hundred examples is only enough when you start from a network someone else trained (lesson 08.9).
- **You must explain every decision.** You can explain a network's prediction afterwards, but you cannot read its rules the way you read a decision tree.
- **Compute is tight.** Training large image or language models needs GPUs. The networks in this module are deliberately small.

It is the right choice for images, audio, video and language, where nobody can write the features, and increasingly for very large tables mixed with text or images.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Add two hand-made features of your own to the digits (for example ink in the top-left quarter). How far can you push accuracy before you give up?
2. Train logistic regression on only the 16 *centre* pixels. How much do the border pixels matter?
3. For each ShopWave task, say "deep learning" or "simpler model" and why: (a) predict next month's spend, (b) read the text of 50,000 reviews for complaints, (c) detect a damaged parcel from a delivery photo, (d) forecast weekly revenue, (e) transcribe phone calls to the support line.

**Quiz** (answers at the end of the notebook)

1. What is the difference between structured and unstructured data? Give a ShopWave example of each.
2. What does "deep" mean in deep learning?
3. Why did six hand-made features score only 68% on digits?
4. Name two situations where you would *not* use deep learning.
5. How many numbers describe one 224×224 colour photo?

## 10. Recap

- Tables come with meaningful columns; images, audio and text do not. Deep learning learns the columns itself.
- Each layer builds more meaningful features from the one below: pixels → edges → parts → objects.
- On digits, learning from raw pixels beat six hand-made features by thirty points (97.8% vs 68%).
- On ShopWave's churn table, a neural network only ties logistic regression. Use deep learning where it earns its cost.
- The rest of this module builds a network from one neuron upwards.

**Next:** [08.2 One neuron, by hand and in NumPy](../02-one-neuron/).
