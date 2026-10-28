# 06.10 · Recommender systems: popularity and collaborative filtering

⏱ about 1 hour · Prerequisites: 01.2 (NumPy arrays), 06.9 (market basket) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's homepage shows the same ten best-sellers to everyone. The e-commerce lead asks: *"Can each customer see products they are actually likely to buy?"*

Market basket rules (06.9) answer "what goes with *this one* product". A recommender answers a personal question: **given everything this customer has bought, what should we show them next?** And it must work for 2,800 customers and 500 products without anyone writing rules.

## 2. The idea

You walk into a shop you visit often, and the assistant says: *"People who liked what you bought last time usually love this."* They are not reading product labels. They are remembering which customers buy which things.

That is **collaborative filtering**. Two products are "similar" if the same customers buy them. To recommend for you, add up how similar each product is to the things you already own, and show the top scorers you do not own yet.

Before anything clever, there is always the simplest recommender: **show the best-sellers**. If your clever system cannot beat that, it is not worth running.

## 3. Definition

- **User-item matrix**: rows are customers, columns are products, a 1 means "bought at least once".
- **Popularity recommender**: rank products by how many customers bought them; recommend the top ones the customer has not bought.
- **Item-item collaborative filtering**: compute similarity between product columns; score each unbought product by the sum of its similarities to the customer's products.
- **User-user collaborative filtering**: find the most similar customers and recommend what they bought.
- **Content-based filtering**: compute similarity from product attributes (description, category, price) instead of purchases.
- **Hit rate @k**: the share of customers for whom at least one of the top-k recommendations is something they actually bought later.
- **Cold start**: a new customer or product with no history to compute with.

## 4. Formula

**Cosine similarity** between two product columns $a$ and $b$:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert} = \frac{\sum_u a_u b_u}{\sqrt{\sum_u a_u^2}\,\sqrt{\sum_u b_u^2}}$$

With 0/1 purchases, the top is "customers who bought both", and each length is √(number of buyers). The result runs from 0 (no buyers in common) to 1 (exactly the same buyers).

**Score** of unbought product $j$ for customer $u$:

$$\text{score}(u, j) = \sum_{i \in \text{bought}(u)} \cos(i, j)$$

## 5. By hand

Four customers, five products:

| | mug | teapot | candle | coaster | lantern |
|---|---|---|---|---|---|
| Ana | 1 | 1 | 0 | 1 | 0 |
| Ben | 1 | 1 | 0 | 0 | 0 |
| Chi | 0 | 1 | 1 | 0 | 1 |
| Dev | 1 | 0 | 0 | 1 | 1 |

**mug vs coaster.** Mug column (1, 1, 0, 1), coaster column (1, 0, 0, 1).
Dot product = 1·1 + 1·0 + 0·0 + 1·1 = **2**. Lengths: √3 = 1.732 and √2 = 1.414.
Cosine = 2 / (1.732 × 1.414) = 2 / 2.449 = **0.816**.

**teapot vs candle.** Teapot (1, 1, 1, 0), candle (0, 0, 1, 0). Dot = 1. Lengths √3 and 1. Cosine = 1 / 1.732 = **0.577**.

**mug vs candle.** No customer bought both: cosine **0**.

The full table:

| | mug | teapot | candle | coaster | lantern |
|---|---|---|---|---|---|
| mug | 1 | 0.667 | 0 | 0.816 | 0.408 |
| teapot | 0.667 | 1 | 0.577 | 0.408 | 0.408 |
| candle | 0 | 0.577 | 1 | 0 | 0.707 |
| coaster | 0.816 | 0.408 | 0 | 1 | 0.500 |
| lantern | 0.408 | 0.408 | 0.707 | 0.500 | 1 |

**Recommend for Ben** (owns mug, teapot). Add the mug and teapot rows for each product he does not own:

- coaster: 0.816 + 0.408 = **1.22**
- lantern: 0.408 + 0.408 = 0.82
- candle: 0 + 0.577 = 0.58

Recommend the **coaster**: the customers who bought mugs also bought coasters.

## 6. In code

```python
from sklearn.metrics.pairwise import cosine_similarity

R = pd.crosstab(sales["Customer ID"], sales["StockCode"]).clip(upper=1)   # customers x products
S = cosine_similarity(R.T.values)              # products x products
np.fill_diagonal(S, 0)
scores = R.values @ S                          # one row of scores per customer
scores[R.values > 0] = -np.inf                 # do not recommend what they own
```

**Evaluate honestly.** Hide each customer's last invoice, learn from everything before it, and check whether the top-10 list contains anything from the hidden invoice.

## 7. On real data

The 500 best-selling products. Training data: 2,805 customers with at least one earlier invoice; only **6.8%** of the matrix cells are 1. That sparsity is normal for recommenders.

**What it learned.** The products most similar to the *white hanging heart T-light holder* are the red hanging heart T-light holder (0.53) and a pink hanging-heart candleholder (0.51). The *Regency cake stand*'s neighbours are Regency teacups in three colours and a Regency teapot. Nobody told the algorithm about product ranges; it found them in baskets.

**Does it beat best-sellers?** Over 2,631 customers whose last invoice contains a top-500 product:

| method | hit rate @10 |
|---|---|
| popularity | 20.3% |
| item-item collaborative filtering | **30.0%** |

The advantage is largest for short lists (exercise 3): at k = 1 collaborative filtering hits 7.2% against 3.5%, about twice as often; at k = 20 the gap narrows to 40.9% vs 32.4%.

**Cold start.** A new customer has no purchases, so every collaborative score is zero. The fallback is the popularity list (Regency cake stand, white hanging heart T-light holder, party bunting…).

## 8. Where it breaks

- **Cold start.** New customers and new products have no history. Fall back to popularity, ask a few questions on sign-up, or use content-based similarity.
- **Popularity bias.** Best-sellers are similar to everything because everyone buys them, so they crowd the lists. Normalising, or down-weighting very popular items, helps.
- **Evaluation offline is not the same as online.** A hit in historical data is not proof a recommendation would *cause* a purchase. A/B tests on the live site are the real measure.
- **Filter bubbles.** Recommending more of the same narrows what customers see. Mixing in some exploration is a business choice.
- **Scale.** A product × product matrix for 100,000 products is 10 billion cells. Real systems use sparse matrices, approximate nearest neighbours, or matrix factorisation.

**Content-based filtering** in one paragraph: describe each product by its attributes (words in the description turned into numbers, category, price band), and recommend products whose description vector is close to those the customer bought. It works for brand-new products and explains itself ("because it is also Regency range"), but it only finds more of the same kind. Many shops combine both. You will build the text side of this in Module 09 with embeddings.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: recommend for Chi (teapot, candle, lantern).
2. User-user filtering on the tiny matrix: who is Ben's nearest customer, and what do you recommend?
3. Hit rate @k for k = 1, 5, 10, 20 on ShopWave. Where is collaborative filtering's advantage biggest?

**Quiz**

1. What is a popularity recommender, and why build it first?
2. What does a cosine similarity of 1 between two products mean? And 0?
3. How does item-item collaborative filtering score a product for a customer?
4. Give one strength of content-based filtering and one of collaborative filtering.
5. What is the cold-start problem, and how do you handle it?

## 10. Recap

- Start with **popularity**: it is simple and surprisingly strong.
- **Item-item collaborative filtering** uses only who bought what: products are similar if the same customers buy them.
- **Cosine similarity** = shared buyers ÷ (√buyers of A × √buyers of B).
- Evaluate by hiding each customer's **last** purchase; on ShopWave, collaborative filtering hits 30% vs 20% at k = 10.
- **Cold start** needs a fallback: popularity or content-based similarity.

**Next:** [06.P Mini-project: RFM customer segments and fraud flags](../project/)
