# 09.10 · Retrieval-augmented generation (RAG)

⏱ about 3 hours · Prerequisites: 09.3 (embeddings, cosine similarity), 09.8, 09.9 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave wants a support assistant on its website that answers questions like *"How long do I have to return a sale item?"* or *"Is there a minimum order for a trade account?"*.

The answers are in ShopWave's help pages: eleven short documents about delivery, returns, refunds, vouchers, wholesale and so on. A language model has never read them. Asked directly, a model either says it cannot know or, worse, writes a typical-sounding policy that is wrong for ShopWave (lesson 09.11 catches one doing exactly that).

You could retrain the model on the help pages (fine-tuning), but that is slow, costs money, and has to be repeated every time a policy changes. There is a simpler way: **find the relevant paragraphs first, and put them in the prompt.**

## 2. The idea

An open-book exam. A student who has to answer from memory may guess; a student allowed to look things up in the course book, and told *"answer only from the book, and say which page"*, gives answers you can check.

RAG gives the model the book. For every question:

1. **Retrieve**: find the few paragraphs of ShopWave's documents most related to the question.
2. **Augment**: paste them into the prompt, with instructions to answer only from them and cite the source.
3. **Generate**: let the model write the answer.

The model's job changes from *remembering* ShopWave's policies (impossible) to *reading and summarising* a few paragraphs (something even small models do well).

## 3. Definition

- **Retrieval-augmented generation (RAG)**: answering a question by first retrieving relevant text from your own documents and then generating an answer grounded in it.
- **Chunk**: a piece of a document, here 80 words, small enough to match one topic and cheap to put in a prompt.
- **Overlap**: words repeated at the start of the next chunk, so a sentence cut at a boundary survives whole in one of them.
- **Embedding**: a vector (here 384 numbers) representing a text's meaning; similar meanings give similar vectors (09.3).
- **Vector database (vector store)**: stores chunks with their embeddings and finds the nearest ones to a query quickly. We use **Chroma**.
- **Top-k**: how many chunks to retrieve per question (here 3).
- **Grounding**: answering only from the retrieved text. **Citation**: naming the source document for each fact.

The full pipeline:

```
documents ─► chunk ─► embed ─► store          (once, offline)
question  ─► embed ─► retrieve top-k chunks ─► prompt = instructions + chunks + question ─► LLM ─► answer + citations
```

## 4. Formula

Retrieval ranks chunks by **cosine similarity** between the question's vector $\mathbf{q}$ and each chunk's vector $\mathbf{c}$:

$$\cos(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{c} \rVert} = \frac{\sum_i q_i c_i}{\sqrt{\sum_i q_i^2}\ \sqrt{\sum_i c_i^2}}$$

1 means the same direction (same meaning), 0 unrelated. Chroma reports a **distance** = 1 − cosine, so the notebook converts it back.

Chunking with size $s$ and overlap $o$ starts a new chunk every $s - o$ words, so a document of $n$ words gives roughly $\lceil (n - o) / (s - o) \rceil$ chunks.

## 5. By hand

**Chunking.** `returns.md` has 140 words. With size 80 and overlap 20, a new chunk starts every 60 words: chunk 1 = words 0–79, chunk 2 = words 60–139. Words 60–79 appear in both, and that is exactly where the sentence about sale items sits, so it is never cut in half. Formula: ⌈(140 − 20) / 60⌉ = 2 chunks.

**Cosine similarity** on three made-up 3-number embeddings:

- q = (0.9, 0.1, 0.3): *"How long do I have to send something back?"*
- a = (0.8, 0.2, 0.4): *"You can return most items within 30 days."*
- b = (0.1, 0.9, 0.2): *"Gift wrap costs £3.50."*

q · a = 0.9×0.8 + 0.1×0.2 + 0.3×0.4 = 0.72 + 0.02 + 0.12 = 0.86
|q| = √(0.81 + 0.01 + 0.09) = √0.91 = 0.954
|a| = √(0.64 + 0.04 + 0.16) = √0.84 = 0.917
cos(q, a) = 0.86 / (0.954 × 0.917) = 0.86 / 0.875 = **0.984**

q · b = 0.09 + 0.09 + 0.06 = 0.24; |b| = √0.86 = 0.927; cos(q, b) = 0.24 / 0.885 = **0.271**

So retrieval returns the returns sentence for the returns question. Real embeddings have 384 numbers, but the arithmetic is the same.

## 6. In code

[lesson.ipynb](lesson.ipynb) builds the whole thing in about 30 lines:

1. Asks `llama3.2` the sale-items question with no documents. In our run it said it did not have ShopWave's policy and suggested checking the website. Honest, but useless to a customer.
2. Loads the 11 documents (1,364 words) and chunks them: **26 chunks**.
3. Embeds with `all-MiniLM-L6-v2` (free, local, 384 numbers per text). The real model agrees with the hand example: the returns question scores **0.670** against the returns sentence and **0.099** against gift wrap, although they share almost no words.
4. Stores the chunks in Chroma and retrieves the top 3 for the sale-items question: both `returns.md` chunks and the Christmas returns page.
5. Generates with the retrieved chunks in the prompt: **"You have 14 days to return a sale item to ShopWave. [returns.md]"**: correct and cited.
6. Compares three questions with and without retrieval. With RAG it gives £3.50 for gift wrap, £150 for a first trade order, and "no customer phone line", each with its source.
7. Asks two questions the documents do not cover (cryptocurrency, furniture assembly): it says **I don't know**, and the best similarity is only 0.23–0.28.

> Outputs in the notebook were produced with `llama3.2` (3B) via Ollama; yours will differ.

## 7. On real data

Eleven documents is a small help centre. Real ones have hundreds of pages, PDFs, and policies that change monthly. The pipeline does not change: chunk, embed and store once (and again when a document changes); retrieve and generate per question. The vector store is what scales: Chroma, pgvector, Pinecone and others search millions of chunks in milliseconds.

## 8. Where it breaks

- **Retrieval misses.** If the right chunk is not in the top-k, the model cannot answer, however good it is. In the exercises a new click-and-collect document is never retrieved for *"Does click and collect cost anything?"*. Always check retrieval separately from generation.
- **Chunks cut facts apart.** Too small and a fact loses its context; too big and a chunk mixes topics and matches vaguely. Overlap helps; measure.
- **The model ignores the context.** Small models sometimes answer from their general knowledge anyway, or cite the wrong document. Instructions help, evaluation catches it (09.11).
- **"I don't know" is a prompt instruction, not a guarantee.** Retrieval always returns *something*; add a similarity threshold for safety.
- **Stale index.** If a policy changes and you do not re-embed it, the assistant confidently quotes the old one.
- **Injected instructions in documents.** Retrieved text goes into the prompt; a document containing "ignore your instructions" is an attack route (09.11).
- **Privacy.** Whatever you retrieve is sent to the LLM provider. With a hosted API, check what you are allowed to send; a local model keeps everything on your machine.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Rebuild the collection with chunk sizes 40/10, 80/20 and 160/40, and count how often the right document is in the top 3 for the 13 answerable evaluation questions.
2. Compute cos(q, b) by hand and check with NumPy.
3. Add a click-and-collect document and ask three questions about it. Diagnose any wrong answer: retrieval or generation?
4. Write a citation checker: does the model ever cite a document it was not given?

**Quiz**

1. Why can't a general LLM answer questions about ShopWave's policies on its own?
2. Name the six steps of the RAG pipeline.
3. Why chunk documents, and why overlap the chunks?
4. Compute cos((0.9, 0.1, 0.3), (0.8, 0.2, 0.4)).
5. Why ask the model to cite its sources?

## 10. Recap

- RAG = **retrieve** relevant chunks, **augment** the prompt with them, **generate** a grounded, cited answer.
- Chunk (80 words, 20 overlap), embed (384-number vectors), store (Chroma), retrieve by **cosine similarity**, generate with instructions to answer only from the context.
- Adding knowledge = adding a document. No retraining.
- Most failures are retrieval failures: check what was retrieved before blaming the model.
- Tell the model to say *I don't know*, and to cite sources so answers can be checked.

**Next:** [09.11 Evaluating LLM systems](../11-evaluating-llms/) turns "it looks good on three questions" into numbers.
