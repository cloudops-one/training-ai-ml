# 09.4 · Attention and the Transformer, explained with a 4-word sentence

⏱ about 2.5 hours · Prerequisites: 09.3, 08.2–08.4 (neurons, softmax, backpropagation), 08.10 (RNNs) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The embedding model in 09.3 and every large language model in 09.7 onwards are **Transformers**. You can use them without knowing how they work, the same way you can drive without knowing how an engine works. But when a model misreads *"not bad at all"*, runs out of memory on a long document, or costs more than expected, the explanation is always in its architecture.

The core of the Transformer is one operation, **self-attention**, and it fits on a sheet of paper. This lesson does it by hand on the sentence *"the mug was broken"*.

## 2. The idea

What does *broken* refer to? To understand a word you need the other words around it. *"The **mug** was broken"* and *"the **promise** was broken"* mean different things, and the difference is in *mug* versus *promise*, three words away.

An RNN (08.10) reads one word at a time and carries a memory forward, like reading through a keyhole. By the time it reaches the end of a long review, the beginning has faded, and it cannot read words in parallel.

**Attention** lets every word look at every other word directly, all at once, and decide how much each one matters to it. Each word asks a question (*"what am I describing?"*), every word offers a label (*"I am an object"*), and each word collects information from the words whose labels best match its question.

## 3. Definition

For each word the model computes three vectors from its embedding, using three learned weight matrices:

- **Query** $q$: what this word is looking for.
- **Key** $k$: what this word offers to others who are looking.
- **Value** $v$: the information this word passes on if attended to.

**Self-attention**: for each word, compare its query with every word's key, turn the scores into weights with softmax, and output the weighted average of the values.

- **Head**: one set of $W_Q, W_K, W_V$. **Multi-head attention** runs several heads side by side, so one head can track *what is broken* while another tracks grammar, then combines them.
- **Positional encoding**: attention itself ignores word order (it is a weighted average), so a vector describing each word's position is added to its embedding first.
- **Transformer layer**: multi-head attention followed by a small feed-forward network, with shortcuts and normalisation. Models stack many layers: DistilBERT has 6, large LLMs have dozens.

## 4. Formula

With all queries, keys and values stacked as rows of matrices $Q, K, V$ and $d_k$ the length of each key:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

and $Q = XW_Q$, $K = XW_K$, $V = XW_V$ where $X$ holds the word embeddings. The division by $\sqrt{d_k}$ keeps the scores from growing with the vector length; without it softmax saturates and puts almost all weight on one word.

Positional encoding in the original Transformer (2017) adds, for position $pos$ and dimension pair $i$:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

a pattern of waves of different lengths, so every position gets a unique signature. Many newer models learn the position vectors instead.

## 5. By hand

The sentence *"the mug was broken"*. Pretend the model has already produced 2-number queries, keys and values (in a real model they come from $XW$):

| word | query $q$ | key $k$ | value $v$ |
|---|---|---|---|
| the | (0, 1) | (1, 0) | (1, 0) |
| mug | (1, 0) | (0, 2) | (0, 4) |
| was | (0, 1) | (1, 0) | (1, 0) |
| broken | (1, 1) | (0, 1) | (2, 2) |

Work out what ***broken*** attends to. $d_k = 2$, so $\sqrt{d_k} = 1.414$.

**Step 1: scores.** Dot product of *broken*'s query (1, 1) with every key, divided by 1.414:

| | the | mug | was | broken |
|---|---|---|---|---|
| $q \cdot k$ | 1 | 2 | 1 | 1 |
| ÷ 1.414 | 0.707 | 1.414 | 0.707 | 0.707 |

**Step 2: softmax.** $e^{0.707} = 2.028$, $e^{1.414} = 4.113$. Sum $= 2.028 \times 3 + 4.113 = 10.197$.

| | the | mug | was | broken |
|---|---|---|---|---|
| weight | 0.199 | **0.403** | 0.199 | 0.199 |

*broken* pays twice as much attention to *mug* as to any other word.

**Step 3: weighted sum of values.**

$$0.199 \times (1, 0) + 0.403 \times (0, 4) + 0.199 \times (1, 0) + 0.199 \times (2, 2) = (0.796,\ 2.01)$$

*broken*'s new vector is mostly made of *mug*'s value. After this layer, the representation of *broken* carries information about *what* is broken. Do the same for every word at once and you have one attention head.

**Decoders cannot look ahead.** A GPT-style model generating text left to right must not see words it has not written yet. It sets the scores for later words to $-\infty$ before the softmax (a **causal mask**), so their weights become 0. For *mug*, only *the* and *mug* remain.

## 6. In code

```python
import numpy as np

scores = Q @ K.T / np.sqrt(K.shape[1])                            # every word against every word
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
output = weights @ V                                               # one row per word
```

Three lines. A real model does this with 64-number vectors, 12 heads per layer and 6 to 100 layers, and learns $W_Q, W_K, W_V$ by backpropagation (08.4).

## 7. On real data

The notebook loads **DistilBERT** (66 million parameters, 6 layers, 12 heads) and asks it for its attention weights on *"the mug arrived broken so i want a refund"*.

- The tokeniser splits *refund* into *ref* + *##und* and adds special tokens [CLS] at the start and [SEP] at the end (09.5).
- **Averaged over heads**, *broken* attends mostly to [CLS] in the first layer (0.29) and to [SEP] in later layers (0.57 in layer 3, 0.67 in layer 6). Special tokens act as a "no-op" parking place when a head has nothing useful to look at.
- **Individual heads specialise.** In layer 5, head 2 puts **0.50** of *broken*'s attention on *mug*, the same thing the hand example did. Only 9 of the 72 heads put more than 0.1 there.

So the real picture is messier than the clean hand example, and that is the honest lesson: attention weights show where information flows, but they are **not an explanation** of the model's decision. 72 heads each do a little, and the meaning emerges from all of them together.

**Three families of Transformer:**

| family | sees | trained to | examples | good for |
|---|---|---|---|---|
| **encoder** | whole text, both directions | fill in hidden words | BERT, DistilBERT, MiniLM | classification, embeddings, search |
| **decoder** | only earlier words (causal mask) | predict the next word | GPT, Claude, Llama | generating text, chat |
| **encoder-decoder** | encoder reads input, decoder writes output | rewrite input into output | T5, BART | translation, summarisation |

## 8. Where it breaks

- **Cost grows with the square of the length.** Every word attends to every word: 1,000 tokens means 1,000,000 scores per head per layer. That is why models have a **context window** limit (09.7) and why long documents are split (09.10).
- **Attention maps are not explanations.** High weight on a word does not prove the model *used* it for its answer.
- **Word order comes only from positional encoding.** Models trained on short texts may handle positions beyond their training length badly.
- **Everything depends on training data.** The architecture is general; what each head learns comes from what the model read.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: compute the attention weights and output for ***mug*** in the table above. Check with NumPy.
2. Multiply all the scores by 10 (as if $d_k$ were large and you forgot to divide by $\sqrt{d_k}$). What happens to *broken*'s weights?
3. By hand: apply a causal mask and recompute *mug*'s weights. Which words can it still see?
4. In DistilBERT, look at the attention from *it* in *"the mug fell off the shelf because it was fragile"*. Which head, if any, links *it* to *mug*?

**Quiz**

1. In one sentence each: what are the query, key and value of a word?
2. Why divide by $\sqrt{d_k}$?
3. Why does a Transformer need positional encoding?
4. Why can GPT-style models not look at later words?
5. A model's context window is 8,000 tokens. Why not simply make it 8 million?

## 10. Recap

- **Self-attention**: $\text{softmax}(QK^\top / \sqrt{d_k})\,V$. Each word's output is a weighted average of all words' values, weighted by how well its query matches their keys.
- By hand on *"the mug was broken"*: *broken* puts weight 0.403 on *mug*, and its output (0.796, 2.01) is mostly *mug*'s value.
- **Multi-head** attention runs many such heads per layer; **positional encoding** adds word order; layers are stacked.
- **Encoders** (BERT) read both ways and suit classification and embeddings; **decoders** (GPT, Claude) use a causal mask and generate text.
- Real attention maps are messy and are not explanations. Cost grows with the square of the text length.

**Next:** [09.5 Hugging Face](../05-hugging-face/): use pretrained Transformers for real tasks in three lines.
