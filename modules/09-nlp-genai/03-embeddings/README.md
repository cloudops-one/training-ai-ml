# 09.3 · Embeddings: meaning as a vector, cosine similarity by hand

⏱ about 2 hours · Prerequisites: 09.1–09.2, 06.6 (PCA), 06.10 (cosine similarity) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A customer types **"tea set"** into ShopWave's search box. The shop sells 3,524 products. A keyword search finds exactly **2** whose names contain "tea set". It misses the *Retro "Tea for One"*, the *Set of Tea Coffee Sugar Tins*, every teapot and teacup. Another customer types **"gift for a gardener"**: keyword search finds **nothing at all**.

The words in a query and the words in a product name are often different words for related things. Bag-of-words (09.1) cannot help: to TF-IDF, *mug* and *cup* are as unrelated as *mug* and *bicycle*. We need numbers that capture **meaning**.

## 2. The idea

Imagine a map where every product sits at a point, and similar products sit close together: teapots near teacups, Christmas lights near Christmas baubles, both far from doormats. To search, put the query on the same map and pick its nearest neighbours.

An **embedding** is that map, except with hundreds of dimensions instead of two. A neural network learns where to put each piece of text by reading enormous amounts of text and being trained so that sentences with similar meaning end up close together. You do not train it; you download it and use it.

## 3. Definition

- **Embedding**: a fixed-length list of numbers (a vector) representing a piece of text. Similar meaning, nearby vectors.
- **Dimension**: the length of the vector. The model used here, `all-MiniLM-L6-v2`, produces **384** numbers for any text up to about 256 word pieces.
- **Sentence embedding**: one vector for a whole sentence or document (as opposed to one per word).
- **Cosine similarity**: how closely two vectors point in the same direction, from −1 (opposite) through 0 (unrelated) to 1 (same direction).
- **Semantic search**: rank documents by the cosine similarity of their embedding to the query's embedding.

The individual 384 numbers have no human meaning. Only the distances between vectors do.

## 4. Formula

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\,\sqrt{\sum_i b_i^2}}$$

If every vector is first scaled to length 1 (**normalised**), the denominator is 1 and cosine similarity is just the dot product. That is why search systems normalise once and then only multiply.

**Word analogies.** Early word embeddings (word2vec, 2013) showed that directions can carry meaning: $\text{king} - \text{man} + \text{woman}$ lands near $\text{queen}$, because the step from *man* to *woman* is roughly the same vector as the step from *king* to *queen*. Modern sentence embeddings are trained differently, but the principle that position encodes meaning is the same.

## 5. By hand

Three made-up 3-D embeddings:

| | dim 1 | dim 2 | dim 3 |
|---|---|---|---|
| mug | 0.9 | 0.1 | 0.2 |
| cup | 0.8 | 0.2 | 0.3 |
| bicycle | 0.1 | 0.9 | 0.1 |

**mug vs cup.**

- dot product: $0.9 \times 0.8 + 0.1 \times 0.2 + 0.2 \times 0.3 = 0.72 + 0.02 + 0.06 = 0.80$
- lengths: $\lVert \text{mug} \rVert = \sqrt{0.81 + 0.01 + 0.04} = \sqrt{0.86} = 0.927$, $\lVert \text{cup} \rVert = \sqrt{0.64 + 0.04 + 0.09} = \sqrt{0.77} = 0.877$
- cosine: $0.80 / (0.927 \times 0.877) = 0.80 / 0.814 = \mathbf{0.983}$

**mug vs bicycle.**

- dot product: $0.09 + 0.09 + 0.02 = 0.20$
- $\lVert \text{bicycle} \rVert = \sqrt{0.83} = 0.911$
- cosine: $0.20 / (0.927 \times 0.911) = \mathbf{0.237}$

Mug and cup point almost the same way; mug and bicycle do not. A real model does exactly this with 384 numbers instead of 3. On the real model, *mug* and *cup* score **0.707**, *mug* and *teapot* 0.511, *mug* and *bicycle* 0.212.

## 6. In code

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # 23 million parameters, ~90 MB
E = model.encode(product_names, normalize_embeddings=True)               # shape (3524, 384)
q = model.encode(["tea set"], normalize_embeddings=True)[0]
scores = E @ q                       # cosine similarity with every product, in one multiplication
best = scores.argsort()[::-1][:5]    # the 5 nearest products
```

## 7. On real data

**Product search.** All 3,524 ShopWave product names (sold at least 5 times) are encoded in a few seconds.

| query | keyword hits | top embedding results (cosine) |
|---|---|---|
| christmas lights | 3 | Christmas Lights 10 Santas (0.761), Christmas Lights 10 Reindeer (0.749), Christmas Lights 10 Vintage Baubles (0.682), Set of 6 T-Lights Santa (0.653) |
| tea set | 2 | Regency Tea Spoon (0.663), Tea Time Tea Set in Gift Box (0.662), Retro "Tea for One" (0.657), Set of Tea Coffee Sugar Tins Pantry (0.654) |
| gift for a gardener | 0 | Botanical Lily Gift Wrap (0.565), Mini Painted Garden Decoration (0.529), Make Your Own Flowerpower Card Kit (0.502) |

Embeddings find related products the keyword search cannot. They are not perfect: a *tea spoon* edges out the actual *tea set*, and "gift for a gardener" returns gift wrap with flowers on it. Embeddings measure **topic similarity**, not "what the customer wants". A real shop combines both (keyword + embedding, called hybrid search) and learns from clicks.

A 2-D PCA picture of the product embeddings shows Christmas items, bags, cards and kitchenware forming separate clouds, without anyone having labelled them.

**Embeddings as features for sentiment.** Encode the 10,000 reviews (6 seconds on the Apple-GPU machine these notebooks were run on, a few minutes on an ordinary laptop CPU) and train logistic regression on the 384 numbers instead of TF-IDF:

| training reviews | embeddings + logistic regression | TF-IDF + logistic regression (09.2) |
|---|---|---|
| 100 | **74.9%** | 63.8% |
| 500 | **81.7%** | 78.0% |
| 2,000 | 84.5% | 84.4% |
| 8,000 | 86.1% | **88.4%** |

With little labelled data, embeddings win by 11 points: the model already knows that *awful* and *terrible* mean the same thing, so it does not need to see both in training. With 8,000 labelled reviews, TF-IDF catches up and overtakes, because it can learn sentiment-specific words that a general-purpose embedding blurs. **Embeddings are the better choice when labels are scarce.** That is exactly the situation of a new shop with few reviews.

## 8. Where it breaks

- **Similar is not the same as relevant.** A tea spoon is "similar" to a tea set. Search needs more than one signal.
- **General-purpose models miss your jargon.** *Retrospot* (a ShopWave pattern name) means nothing to a model trained on the web.
- **Length limit.** `all-MiniLM-L6-v2` reads at most 256 word pieces and silently ignores the rest. Here only 0.35% of reviews are longer, but long documents must be split (09.10).
- **Negation is still hard.** *"Not bad at all, I'm pleasantly surprised"* scores 0.26 positive on embeddings too.
- **The numbers are not explainable** the way TF-IDF coefficients are. You cannot point at dimension 117 and say what it means.
- **Embeddings inherit biases** from the text they were trained on.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: cosine similarity of $(1, 2, 2)$ and $(2, 1, 2)$. Then normalise both to length 1 and check that their dot product gives the same answer.
2. Try five queries of your own on the product search. Where does it beat keywords, and where does it fail?
3. Find near-duplicate product names: pairs with cosine above 0.95. Are they the same product written twice, or different products? Why would a data team care?
4. Search the test reviews for the 3 most similar to *"it arrived broken and I want my money back"*.

**Quiz**

1. What does one number inside an embedding vector mean?
2. Why do search systems normalise embeddings to length 1?
3. *mug* and *cup* have TF-IDF similarity 0 but embedding similarity 0.71. Why?
4. With 100 labelled reviews, embeddings beat TF-IDF by 11 points. With 8,000, TF-IDF wins. Explain both.
5. Give one reason not to rely on embedding search alone in a shop.

## 10. Recap

- An **embedding** turns text into a fixed-length vector (384 numbers here) so that similar meanings are close.
- **Cosine similarity** $= \dfrac{\mathbf a \cdot \mathbf b}{\lVert \mathbf a\rVert \lVert \mathbf b\rVert}$; by hand, mug–cup 0.983, mug–bicycle 0.237. With normalised vectors it is a dot product.
- Semantic search finds *Retro "Tea for One"* for "tea set" and garden items for "gift for a gardener", where keywords find 2 and 0 results; but similar is not always relevant.
- As classifier features, embeddings win with little labelled data (74.9% vs 63.8% on 100 reviews), TF-IDF wins with plenty (88.4% vs 86.1% on 8,000).

**Next:** [09.4 Attention and the Transformer](../04-attention-transformers/) opens up the kind of network that produces these embeddings.
