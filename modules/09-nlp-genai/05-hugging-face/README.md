# 09.5 · Hugging Face: pretrained models in three lines

⏱ about 1.5 hours · Prerequisites: 09.2–09.4 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's support inbox gets 300 emails a day. Someone sorts them by hand into *delivery*, *refund*, *product question* and *complaint* before anyone can answer. There are no labelled emails to train a classifier on, and nobody has time to label a few thousand.

Meanwhile, thousands of organisations have already trained Transformer models for sentiment, topic classification, finding names in text and much more, and published them for free. Training a model like that from scratch costs days of GPU time. Downloading one costs a minute.

## 2. The idea

**Hugging Face** is the app store of machine-learning models: a public hub with hundreds of thousands of trained models and a Python library, `transformers`, that downloads and runs any of them. A **pipeline** bundles everything a model needs (its tokeniser, the network, and the code that turns its raw output into labels and scores) behind one function call:

```python
pipeline("sentiment-analysis")("The mug arrived chipped.")   # → NEGATIVE, 0.99
```

For the support inbox there is a clever trick: **zero-shot classification**. A model trained to judge whether one sentence implies another can sort text into *any* labels you invent, without seeing a single labelled example.

## 3. Definition

- **Model hub**: an online library of trained models, each with a name such as `distilbert/distilbert-base-uncased`. A trained model is also called a **checkpoint**.
- **Pipeline**: tokeniser + model + post-processing for one task.
- **Tokeniser**: splits text into the word pieces the model was trained on and maps them to numbers (**token ids**). Unknown words are split into known pieces (*retrospot* → *retro*, *##sp*, *##ot*). Special tokens such as [CLS] and [SEP] mark the start and end.
- **Zero-shot classification**: classify into labels never used in training, here by asking a natural-language-inference model whether the text implies *"This example is {label}."*
- **Named-entity recognition (NER)**: tagging people, places and organisations in text.
- **Fill-mask**: guessing a hidden word, the task BERT-style models are pre-trained on.

## 4. Formula

**Zero-shot scoring.** For an email $x$ and candidate labels $\ell_1, \dots, \ell_k$, the inference model returns a logit $s_j$ for *"$x$ implies 'This example is $\ell_j$.'"*. Softmax across the labels turns them into scores that sum to 1:

$$P(\ell_j \mid x) = \frac{e^{s_j}}{\sum_{i=1}^{k} e^{s_i}}$$

**Memory of a model.**

$$\text{memory} \approx \text{parameters} \times \text{bytes per parameter}$$

with 4 bytes per parameter in float32, 2 in float16, and about 0.5 in the 4-bit formats used for local LLMs.

## 5. By hand

**How much memory?**

| model | parameters | float32 | float16 |
|---|---|---|---|
| DistilBERT (sentiment, zero-shot) | 67 million | 67M × 4 = **268 MB** | 134 MB |
| BERT-base (NER) | 108 million | 108M × 4 = **431 MB** | 215 MB |
| a 3-billion-parameter LLM | 3 billion | **12 GB** | 6 GB |

The models in this lesson fit on any laptop. A 3B LLM in float32 would not fit in 8 GB of RAM at all; at 4 bits it needs about 1.5 GB. That arithmetic is behind the model-for-your-memory table in the [setup guide](../../../setup/README.md#running-a-model-on-your-own-computer-with-ollama).

**Reading a tokeniser.** *"ShopWave's retrospot mugs arrived unbroken!"* becomes 12 pieces: *shop, ##wave, ', s, retro, ##sp, ##ot, mug, ##s, arrived, unbroken, !*, then wrapped as ids `101 … 102` ([CLS] … [SEP]). Common words are one piece; ShopWave's own words are several.

## 6. In code

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
router = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
router(email, candidate_labels=["delivery", "refund", "product question", "complaint"])
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
fill = pipeline("fill-mask", model="distilbert/distilbert-base-uncased")
```

Always name the model explicitly. Defaults change between library versions, and a pipeline without a model name may silently switch models. Pass `device="mps"` (Mac) or `device=0` (NVIDIA GPU) to use a GPU; the notebook picks automatically.

## 7. On real data

- **Sentiment.** A DistilBERT model fine-tuned on **film** reviews (the SST-2 dataset) scores **89.0%** on our 2,000 test product reviews, having never seen one of them. That beats the best TF-IDF model trained on 8,000 of our reviews (88.4%, 09.2). It also reads *"Not bad at all, I'm pleasantly surprised"* as positive (0.9998), which TF-IDF and embeddings got wrong.
- **Routing support emails.** The zero-shot model sorts **6 of 8** sample emails correctly with no training data. It reads *"my order still hasn't arrived"* as a complaint and a refund request that explains why the lantern does not fit as a product question.
- **NER** finds *Sarah Jones* (person), *Paris* and *London* (places) and *ShopWave* (organisation), and tags the *Kew Gardens* product range as a place with lower confidence (0.67).
- **Fill-mask.** *"The parcel arrived [MASK] and the mug was broken."* → *safely* (0.13), *promptly*, *quickly*. The model fills in the most likely word in general text, not the fact the sentence implies.
- **Summarisation** used to be a pipeline too. In current versions of `transformers` it has been removed in favour of general text-generation models; you will summarise with an LLM in 09.8.

## 8. Where it breaks

- **Somebody else's training data.** The sentiment model learned from film reviews; it works on product reviews because language is shared, but domain-specific words (*retrospot*, *chipped*) may confuse it. Test on your own data before trusting it.
- **Zero-shot is sensitive to wording.** Changing the hypothesis template from *"This example is {}."* to *"This customer email is about {}."* drops the router from 8 to 3 correct out of 10 in the exercises. Small wording changes, big effects: the start of prompt engineering (09.8).
- **Bigger is not automatically better.** A model nearly three times the size gets 7 of 8 instead of 6; a common 1.6 GB default did worse on these emails. Only labelled examples tell you.
- **Licences differ.** Every model on the hub has its own licence; check it before using a model commercially.
- **Downloads and memory add up.** Each model is hundreds of megabytes; a notebook using four of them needs a gigabyte of disk and RAM.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Memory by hand: DistilBERT, BERT-base, a 3B and a 7B LLM, in float32 and float16. Which fit in 8 GB?
2. Add a *praise* topic and two thank-you emails. Then change the hypothesis template. What happens to accuracy?
3. Route the emails with `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`. Is the bigger model worth it?
4. Tokenise ten ShopWave product names. Which get split into the most pieces, and why does it matter?

**Quiz**

1. What three things does a pipeline bundle together?
2. Why must a model always be used with its own tokeniser?
3. How can a zero-shot classifier sort emails into labels it was never trained on?
4. Roughly how much memory does a 66-million-parameter model need in float32?
5. Why does the fill-mask model say the parcel arrived *safely* when the mug was broken?

## 10. Recap

- The Hugging Face hub hosts hundreds of thousands of trained models; `pipeline(task, model=…)` runs one in a line.
- A sentiment model trained on film reviews scores 89.0% on our product reviews without any training by us.
- **Zero-shot classification** routes support emails into labels you invent (6 of 8 correct here), but is sensitive to wording.
- Tokenisers split rare words into pieces; memory ≈ parameters × bytes per parameter.
- Pretrained is a starting point, not a guarantee: always evaluate on your own labelled examples.

**Next:** [09.6 Fine-tuning](../06-fine-tuning/): adapt a pretrained model to ShopWave's own data.
