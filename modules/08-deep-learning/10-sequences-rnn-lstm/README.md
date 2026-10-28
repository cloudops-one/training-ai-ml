# 08.10 · Sequences: RNNs, LSTMs and why Transformers replaced them

⏱ about 1.5 hours · Prerequisites: [08.5](../05-first-keras-network/), [08.6](../06-training-well/) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Two ShopWave reviews use exactly the same words:

- *"Not good. The quality is excellent, but delivery was slow."*
- *"Good. The quality is not excellent, but delivery was slow."*

A model that only counts which words appear sees the same thing twice. So does every model you have built so far: tables (Modules 02–07) have no order, and a CNN (08.8) looks at a fixed-size picture. Yet a lot of business data is a **sequence**: words in a review, clicks in a browsing session, a customer's orders over time. In a sequence, *what came before* changes what something means.

This lesson builds the classic tool for sequences, the **recurrent neural network**, by hand. It also shows its weakness, and why the Transformers behind modern language models (Module 09) replaced it.

## 2. The idea

You read a sentence one word at a time, carrying a running sense of what it is about. When you reach "slow", you still remember that "not good" came first.

An RNN does the same. It reads one item at a time and keeps a small **memory** (the hidden state). At each step it mixes the new item with its memory to make a new memory. The same rule, with the same weights, is applied at every step, just as a CNN reuses one filter at every position.

The trouble is that the memory fades. Information from many steps back is multiplied by the same small factor again and again until almost nothing is left. An **LSTM** adds gates, small learned switches that decide what to keep, what to forget and what to output, so important information can survive much longer.

## 3. Definition

- **Sequence**: ordered inputs $x_1, x_2, \dots, x_T$.
- **Hidden state** $h_t$: the RNN's memory after reading $x_t$.
- **Recurrent neural network (RNN)**: a layer that updates $h_t$ from $x_t$ and $h_{t-1}$ with the same weights at every step. Keras: `SimpleRNN`.
- **Vanishing gradient**: during training, the error signal is multiplied by a factor at each step backwards; with factors below 1 it shrinks towards zero, so the network cannot learn from items far back.
- **LSTM (long short-term memory)**: an RNN cell with a separate *cell state* $c_t$ and three gates:
  - **forget gate**: how much of the old memory to keep;
  - **input gate**: how much of the new information to write in;
  - **output gate**: how much of the memory to reveal as $h_t$.
- **GRU**: a simpler variant with two gates; usually performs like an LSTM.
- **Embedding layer**: turns each word number into a short learned vector (Module 09 explains embeddings properly).

## 4. Formula

The simple RNN, one step:

$$h_t = \tanh(w_x \, x_t + w_h \, h_{t-1} + b), \qquad h_0 = 0$$

With vectors, $w_x$ and $w_h$ become matrices, but the shape of the rule is the same.

The LSTM in words (each gate is a sigmoid between 0 and 1, computed from $x_t$ and $h_{t-1}$):

```
            ┌──────────── cell state c (long-term memory) ─────────────┐
c_{t-1} ──► × forget gate ──► + ◄── input gate × new candidate ──────► c_t
                                                                        │
                                              output gate × tanh(c_t) ──► h_t
```

$$c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t \qquad h_t = o_t \cdot \tanh(c_t)$$

The key is the **+**: memory is carried forward by addition, not repeated multiplication, so it does not have to fade.

Vanishing: a signal multiplied by a factor $r$ for $n$ steps becomes $r^n$.

## 5. By hand

Take $w_x = 0.5$, $w_h = 0.8$, $b = 0$ and the sequence $x = [1, 0, -1]$.

| step | $x_t$ | inside tanh | $h_t$ |
|---|---|---|---|
| 1 | 1 | 0.5·1 + 0.8·0 = 0.5 | tanh(0.5) = **0.4621** |
| 2 | 0 | 0.5·0 + 0.8·0.4621 = 0.3697 | tanh(0.3697) = **0.3537** |
| 3 | −1 | 0.5·(−1) + 0.8·0.3537 = −0.2170 | tanh(−0.2170) = **−0.2137** |

Now the same numbers in reverse order, $x = [-1, 0, 1]$:

| step | $x_t$ | inside tanh | $h_t$ |
|---|---|---|---|
| 1 | −1 | −0.5 | **−0.4621** |
| 2 | 0 | 0.8·(−0.4621) = −0.3697 | **−0.3537** |
| 3 | 1 | 0.5 + 0.8·(−0.3537) = 0.2170 | **0.2137** |

Same inputs, different order, opposite final memory. A bag-of-words model would have given the same answer for both.

**Vanishing.** If each step back multiplies the error signal by 0.5, after 20 steps:

$$0.5^{20} = 0.00000095$$

Even a factor of 0.9 leaves only $0.9^{20} = 0.12$. A factor of 1.1 explodes to $1.1^{20} = 6.73$. In the notebook, a single input followed by 19 zeros leaves $h$ at 0.4621 after step 1, 0.0555 after step 10 and 0.0059 after step 20.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). It checks the table with a five-line NumPy RNN and with a Keras `SimpleRNN` whose weights are set by hand (both give 0.4621, 0.3537, −0.2137).

A sequence model in Keras is an embedding followed by a recurrent layer:

```python
model = keras.Sequential([
    keras.Input((200,)),                 # 200 word numbers per review
    layers.Embedding(10_000, 32),        # each word → 32 learned numbers
    layers.LSTM(32),                     # read them in order; keep 32 numbers of memory
    layers.Dense(1, activation="sigmoid"),
])
```

Swap `layers.LSTM` for `layers.SimpleRNN` or `layers.GRU` to change the cell, or for `layers.GlobalAveragePooling1D()` to get a "bag of words" model that ignores order.

## 7. On real data

**A task where order is everything.** Sequences of 20 random tokens contain one `1` and one `2`; the label is whether the `1` comes first. The bag model, which averages the tokens, scores **0.521**: a coin flip, because the information it needs is not in the counts. The LSTM scores **1.000**.

**IMDB film reviews** (Keras' built-in sentiment dataset; 10,000 reviews to train, 5,000 to test, last 200 words of each):

| model | weights | test accuracy |
|---|---|---|
| bag of words (average embedding) | 320,033 | 0.835 |
| SimpleRNN | 322,113 | 0.784 |
| LSTM | 328,353 | 0.836 |

The LSTM beats the plain RNN but only ties the bag model. Shuffling the words inside every test review shows why: the bag model's accuracy does not move (0.835), the LSTM's drops to 0.740. The LSTM *does* use word order, but for "is this review positive?" the words themselves carry almost all of the signal.

In the exercises, at 100 tokens the order task defeats the SimpleRNN (0.700) while the LSTM still scores 0.949: vanishing memory, measured.

## 8. Where it breaks

- **Slow.** An RNN must read step 1 before step 2. It cannot process a whole sentence at once on a GPU, so training is slow on long texts.
- **Still forgets.** LSTMs remember much longer than plain RNNs, but hundreds of steps back is still hard.
- **Order is not always the point.** As the IMDB result shows, a simpler model that ignores order can do just as well. Always compare with a baseline.
- **Fixed vocabulary.** Words outside the 10,000 most common are dropped. Module 09 uses sub-word tokens instead.
- **Forecasting.** RNNs can forecast time series, but on ShopWave's two years of sales, simple seasonal methods and gradient boosting (Module 07) are hard to beat. Deep sequence models need far more data.

**Why Transformers took over.** In 2017 the Transformer replaced recurrence with **attention**: every word looks directly at every other word in one step, however far apart. Nothing has to survive a long chain of multiplications, and all words are processed in parallel, which suits GPUs. Every large language model you will use in Module 09 is a Transformer.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: how much of a single input survives five steps with $w_h = 0.2$, and with $w_h = 0.95$?
2. Train a GRU on the order task and compare it with the LSTM.
3. Make the order task 100 tokens long. Compare SimpleRNN and LSTM.
4. Train the IMDB LSTM on only the last 50 words of each review. What changes?

**Quiz**

1. What does the hidden state $h_t$ hold?
2. Which weights does an RNN use at step 5 compared with step 1?
3. What is $0.5^{20}$, and what does it mean for training an RNN?
4. Name the three gates of an LSTM and what each does.
5. Give two reasons Transformers replaced RNNs.

## 10. Recap

- Sequences carry meaning in their order; tables and bag-of-words models throw it away.
- An RNN reads one step at a time: $h_t = \tanh(w_x x_t + w_h h_{t-1} + b)$, with the same weights at every step.
- Its memory, and its learning signal, fade over long sequences: $0.5^{20} \approx 10^{-6}$.
- LSTMs (and GRUs) add gates and an additive memory path, so information survives longer: 0.949 against 0.700 on a 100-step task.
- On sentiment, word order adds little (bag 0.835, LSTM 0.836). Transformers, which replaced RNNs, are the subject of Module 09.

**Next:** [08.11 The same model in PyTorch](../11-pytorch/) (optional), then the [module project](../project/).
