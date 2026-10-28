# Module 09 · Language & Generative AI

**ShopWave asks:** *"We get 50,000 reviews a year and 300 support emails a day. Read them for us, and answer the easy ones."*

**Time:** about 24 hours. **Prerequisites:** Module 03 (classification metrics), Module 04 (honest evaluation), Module 06 (cosine similarity, PCA), Module 08 (neural networks, training loops).

## Where you are

Every model so far has eaten numbers: visits, spend, days since the last order. ShopWave's most valuable data is not numbers. It is what customers *write*: reviews that say why they love the fairy lights, emails that say the cake stand arrived in pieces, questions about whether a teapot is dishwasher safe. Nobody at ShopWave has time to read them all.

This module turns text into something a model can use, in three generations of technique:

1. **Counting words** (09.1–09.2). Bag-of-words and TF-IDF. Fast, cheap, explainable, and still a strong baseline.
2. **Learned meaning** (09.3–09.6). Embeddings that place similar meanings close together, the Transformer architecture behind every modern language model, ready-made models from Hugging Face, and fine-tuning one on your own data.
3. **Large language models** (09.7–09.12). How LLMs work, prompting them, calling them from Python, grounding them in your own documents with retrieval (RAG), checking their answers, and letting them use tools.

You will finish with a review analyser and a support chatbot that answers from ShopWave's own policies.

## What you will be able to do afterwards

- Turn text into numbers by hand (bag-of-words, TF-IDF) and in scikit-learn, and know what is lost on the way.
- Build and honestly evaluate a sentiment classifier, and explain its mistakes.
- Use embeddings for meaning-based search, and compute cosine similarity by hand.
- Explain self-attention with a worked example, and the difference between BERT-style and GPT-style models.
- Use pretrained models from Hugging Face, and fine-tune one on your own labelled data.
- Explain how an LLM is trained and generates text, prompt it well, get structured output from it, and call it from Python with any provider.
- Build a retrieval-augmented chatbot, evaluate it, and defend it against hallucination and prompt injection.
- Give an LLM tools and understand what makes an agent.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 09.1 | [Text to numbers: cleaning, tokens, bag-of-words, TF-IDF by hand](01-text-to-numbers/) | 2 h |
| 09.2 | [Sentiment classifier on product reviews](02-sentiment-classifier/) | 1.5 h |
| 09.3 | [Embeddings: meaning as a vector, cosine similarity by hand](03-embeddings/) | 2 h |
| 09.4 | [Attention and the Transformer, explained with a 4-word sentence](04-attention-transformers/) | 2.5 h |
| 09.5 | [Hugging Face: pretrained models in three lines](05-hugging-face/) | 1.5 h |
| 09.6 | [Fine-tuning a small model on ShopWave reviews](06-fine-tuning/) | 2 h |
| 09.7 | [How large language models work](07-how-llms-work/) | 1.5 h |
| 09.8 | [Prompt engineering and structured output](08-prompt-engineering/) | 2 h |
| 09.9 | [Calling an LLM from Python: APIs, cost, safety, local models](09-calling-llms-from-python/) | 1.5 h |
| 09.10 | [Retrieval-augmented generation: chunk, embed, retrieve, generate](10-rag/) | 3 h |
| 09.11 | [Evaluating LLM outputs; hallucination, bias, prompt injection](11-evaluating-llms/) | 1 h |
| 09.12 | [Agents and tool calling](12-agents-tool-calling/) | 1.5 h |
| 09.P | [Mini-project: review analyser and ShopWave support chatbot](project/) | 2 h |

## The scoreboard you will build (09.2–09.6)

Every sentiment method is tested on the same 2,000 held-out reviews:

| method | trained on | test accuracy | lesson |
|---|---|---|---|
| guess the majority class | nothing | 50.0% | 09.2 |
| TF-IDF + logistic regression | 2,000 reviews | 84.4% | 09.3 |
| embeddings + logistic regression | 8,000 reviews | 86.1% | 09.3 |
| TF-IDF (words and word pairs) + logistic regression | 8,000 reviews | 88.4% | 09.2 |
| ready-made DistilBERT sentiment model (trained on film reviews) | none by us | 89.0% | 09.5 |
| DistilBERT fine-tuned by you | 2,000 reviews | 90.6% | 09.6 |

The pattern is the lesson: newer methods win, but by less than the hype suggests, and the old method on more data is closer than you would think. With only 100 labelled reviews, though, embeddings beat TF-IDF by 11 points (09.3). Which tool is right depends on how much labelled data you have.

## Data

- **Reviews.** ShopWave has no review data of its own, so the course uses 10,000 real Amazon product reviews (books, music, kitchenware, electronics and more) from the public **amazon_polarity** dataset (Zhang, Zhao and LeCun, 2015; Apache 2.0). Half are negative (1–2 stars), half positive (4–5 stars). `load_reviews()` in `data/download.py` downloads one 117 MB file on first use and caches the sample. Treat them as ShopWave's reviews.
- **Products.** Real ShopWave product names from Online Retail II (`load_retail()`), used for meaning-based product search in 09.3.
- **Support emails and policies.** Short examples written for the course (09.5 onwards).

## Which LLM? You choose

Lessons 09.7–09.12 call a large language model. Every notebook goes through one small helper, [`lib/course_llm.py`](../../lib/course_llm.py), so **the same notebook works with any provider**: a free model on your own computer with Ollama, Claude, OpenAI, Gemini, Groq or LM Studio. You pick in a `.env` file:

```
LLM_PROVIDER=ollama          # or anthropic, openai, gemini, groq, lmstudio
LLM_MODEL=llama3.2           # optional
```

No API key? Use Ollama, it is free. The [setup guide](../../setup/README.md#large-language-models-for-module-09) has install steps for Windows, macOS and Linux and a table of **which model to choose for 8 GB, 16 GB or more memory**. Run `self_test()` once to check your choice works.

Lessons 09.1–09.6 need no LLM at all. They download a few small models from Hugging Face (about 1.5 GB in total, cached outside the course folder) and run on any laptop CPU.

## The maths in this module

Counting and logarithms (TF-IDF), the dot product and cosine similarity (from 06.10), softmax (from 03.8 and 08.3), and matrix multiplication for attention. Every formula is computed by hand on a handful of numbers before it is used.
