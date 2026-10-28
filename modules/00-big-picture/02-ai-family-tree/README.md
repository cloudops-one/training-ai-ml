# 00.2 · The AI family tree: AI, ML, deep learning, generative AI

⏱ about 30 minutes · Prerequisites: 00.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The ShopWave CEO forwards you three emails in one morning:

1. A vendor selling "AI-powered fraud detection".
2. A consultant proposing "a deep learning model for demand forecasting".
3. A newsletter about "generative AI for customer support".

Then the question: *"Are these the same thing? Which one do we need?"*

If you cannot place these words on one map, you cannot answer, and you cannot tell marketing from substance. This lesson gives you the map. It is short, and you will use it every day.

## 2. The idea

The four terms are not rivals. They are **nested boxes**, each inside the one before:

```
┌─────────────────────────────────────────────────────────────┐
│  ARTIFICIAL INTELLIGENCE                                    │
│  any program that does something we'd call "smart"          │
│  (rules, search, planning, ... and learning)                │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  MACHINE LEARNING                                   │   │
│   │  programs that learn the rules from data            │   │
│   │  (linear regression, decision trees, k-means, ...)  │   │
│   │                                                     │   │
│   │   ┌─────────────────────────────────────────────┐   │   │
│   │   │  DEEP LEARNING                              │   │   │
│   │   │  ML using neural networks with many layers  │   │   │
│   │   │  (image recognition, speech, translation)   │   │   │
│   │   │                                             │   │   │
│   │   │   ┌─────────────────────────────────────┐   │   │   │
│   │   │   │  GENERATIVE AI                      │   │   │   │
│   │   │   │  deep models that create content    │   │   │   │
│   │   │   │  (LLMs like Claude/GPT, image gen)  │   │   │   │
│   │   │   └─────────────────────────────────────┘   │   │   │
│   │   └─────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Everything in an inner box is also in the outer boxes. A large language model is generative AI, which is deep learning, which is machine learning, which is AI. A decision tree is machine learning and AI, but not deep learning. A hand-written fraud rule is AI (arguably) but not machine learning at all.

## 3. Definition

- **Artificial intelligence (AI)**: the broad goal of making computers do tasks that need intelligence when humans do them. It includes rule-based systems, search algorithms, planning, and learning. The term dates from 1956.
- **Machine learning (ML)**: the part of AI where the program learns the rules from data, as you saw in 00.1. Most of this course lives here.
- **Deep learning (DL)**: machine learning done with **neural networks** that have many layers. A neural network is a model built from thousands or millions of tiny units (neurons), each doing a weighted sum and a squash. "Deep" just means many layers stacked. Deep learning is what made computers good at images, speech and language after about 2012. Module 08.
- **Generative AI (GenAI)**: deep learning models trained to *produce* new content rather than only label or score existing content. Large language models (LLMs) generate text; diffusion models generate images. Module 09.

Two more words you will hear in the same breath:

- **Data science**: the broader job of extracting insight from data: statistics, visualisation, ML, communication. ML is a tool inside it.
- **Predictive analytics** / **AutoML** / **cognitive computing**: marketing names for ML. When a vendor uses them, ask: *what data does it learn from, and what does it predict?*

## 4. Formula

No formula in this lesson. Instead, one test that sorts any product or proposal:

> **"What does it learn from, and what does it output?"**

| Learns from | Outputs | It is |
|---|---|---|
| nothing (rules written by people) | a decision | AI, not ML |
| a table of examples with answers | a number or label | ML (classical) |
| images, audio, raw text, in bulk | a number or label | deep learning |
| huge text or image corpora | new text, images, code | generative AI |

## 5. By hand

Sort the CEO's three emails and seven more ShopWave ideas into the right box. Cover the answer column first.

| # | Proposal | Box |
|---|---|---|
| 1 | Vendor's "AI fraud detection" that trains on our past orders and flags new ones | ML (probably a tree ensemble; could be DL, ask) |
| 2 | "Deep learning for demand forecasting" of monthly sales from a 2-year table | ML is enough; DL is overkill for a small table |
| 3 | "Generative AI for customer support" that drafts replies | Generative AI (an LLM) |
| 4 | A checkout rule: free shipping over £50 | AI at most; plain code, no learning |
| 5 | Auto-tagging 50,000 product photos as mug / candle / cushion | Deep learning (images) |
| 6 | Grouping customers into segments for marketing | ML (clustering, unsupervised) |
| 7 | A chatbot that answers "where is my order?" from our policy documents | Generative AI + retrieval (RAG) |
| 8 | Predicting next month's spend per customer from their history | ML (regression) |
| 9 | Converting voice messages from customers into text | Deep learning (speech) |
| 10 | A spreadsheet formula that computes VAT | Not AI at all |

Notice row 2: deep learning is not "better ML". For a small table of numbers, classical ML usually wins, trains in seconds, and can be explained. Deep learning earns its cost on images, audio and text. You will test this yourself in Module 08.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). It runs one example from each box on tiny data so you see the *shape* of each:

1. A rule (plain `if`).
2. Classical ML: a decision tree on a 10-row table.
3. Deep learning: a tiny neural network on the same table, so you can see it is just another model that takes `X` and `y`.
4. Generative AI: a call to a language model, with a local fallback if you have no API key.

You are not expected to understand steps 3 and 4 yet. Look at how similar the code is: `fit`, `predict`, or `prompt in, text out`.

## 7. On real data

The notebook ends by asking: of ShopWave's real columns (invoice, product, quantity, price, customer, country, date), which problems are classical ML and which would need deep learning? Answer: almost everything about this table is classical ML. Deep learning enters only when ShopWave adds photos, reviews or support chats, which is Modules 08 and 09.

## 8. Where it breaks

- **"AI" on a label tells you nothing.** Ask what it learns from and what it outputs.
- **Deeper is not better.** Neural networks need lots of data, compute and tuning. On tabular business data, gradient-boosted trees (Module 05) usually beat them.
- **Generative AI is not a database.** An LLM produces fluent text, not verified facts. For "where is my order?" you combine it with retrieval from your own records (Module 09).
- **The boxes blur at the edges.** Some systems mix rules and learning; an LLM can be used as a classifier. The map is for orientation, not for arguments.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Find three products or news headlines that use the word "AI". For each, answer "learns from what, outputs what" and place it in a box.
2. In the notebook, the tree and the neural network reach similar accuracy on 10 rows. Change the data to 10,000 rows (a cell is provided). Which trains faster? Which is easier to read?
3. Write one ShopWave problem for each of the four boxes that is *not* in the table above.

**Quiz**

1. Which of these is *not* machine learning: a decision tree, a neural network, a VAT formula, k-means?
2. Is every deep learning model a machine learning model? Is every ML model a deep learning model?
3. Name the single question that sorts any "AI product" into a box.
4. ShopWave has a 2-year table of sales. A consultant proposes deep learning. What is a reasonable reply?
5. What does "deep" in deep learning refer to?

## 10. Recap

- **AI ⊃ ML ⊃ DL ⊃ GenAI**: nested boxes, not competitors.
- ML learns rules from data; DL is ML with many-layered neural networks; GenAI is DL that creates content.
- Sort any proposal with *"learns from what, outputs what?"*
- Tables of business numbers → classical ML. Images, audio, text → deep learning. Producing text or images → generative AI.
- "AI" in a product name is marketing until you know the data and the output.

**Next:** [00.3 The three kinds of learning](../03-three-kinds-of-learning/) splits machine learning itself into three, by one question: *do we already have the answers?*
