# 09.6 · Fine-tuning a small model on ShopWave reviews

⏱ about 2 hours · Prerequisites: 09.5, 08.4–08.6 (backpropagation, training loops), 08.9 (transfer learning) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The ready-made sentiment model from 09.5 scores 89.0% on ShopWave's reviews, but it learned from film reviews and has never seen a product complaint. The zero-shot router is more flexible but less accurate. ShopWave does have something those models never had: **2,000 of its own reviews, already labelled**.

Training a Transformer from scratch would need billions of words and weeks of GPU time. But you do not have to start from scratch. You can take a model that already understands English and teach it the one extra thing you need, in seconds to minutes. That is **fine-tuning**.

## 2. The idea

Hiring a new support agent, you do not raise a baby and teach them to read. You hire an adult who already reads fluently and give them a two-week induction on ShopWave's products and tone. Fine-tuning is the induction.

DistilBERT was **pre-trained** on books and Wikipedia to fill in missing words (09.5). In doing so it learned grammar, word meanings and a great deal about the world. Fine-tuning bolts a small new **classification head** on top, then trains the whole network a little further on your labelled reviews, with a small learning rate so the existing knowledge is adjusted, not erased.

## 3. Definition

- **Pre-training**: the expensive first training on huge unlabelled text, done once by someone else.
- **Fine-tuning**: continued training of a pre-trained model on a small labelled dataset for a specific task. A form of **transfer learning** (08.9).
- **Classification head**: the new layers that turn the model's summary of the text (the [CLS] vector) into one score per class. Here 592,130 parameters, 0.9% of the model.
- **Step / batch / epoch**: one weight update / the reviews used in it (16 here) / one full pass over the training data (125 steps for 2,000 reviews).
- **Learning rate**: how big each update is. Fine-tuning uses tiny ones ($5 \times 10^{-5}$) compared with training from scratch.
- **Parameter-efficient fine-tuning (LoRA and similar)**: freezing the original weights and training a small add-on instead. Common for large language models; not needed for a model this small.

## 4. Formula

The model outputs two numbers (**logits**) $z_{\text{neg}}, z_{\text{pos}}$ for each review. Softmax turns them into probabilities, and the **cross-entropy loss** is the negative log of the probability given to the true class:

$$p_{\text{pos}} = \frac{e^{z_{\text{pos}}}}{e^{z_{\text{neg}}} + e^{z_{\text{pos}}}} \qquad L = -\ln p_{\text{true}}$$

Each step averages $L$ over a batch, backpropagation (08.4) computes the gradient for all 67 million weights, and the AdamW optimiser moves each weight a little against its gradient.

## 5. By hand

**Where does training start?** The head is random, so the model's guesses are roughly 50/50. The loss for a 50/50 guess is $-\ln 0.5 = \ln 2 = \mathbf{0.693}$, whatever the true class.

**Two predictions for a positive review:**

| logits (neg, pos) | $e^{z}$ | $p_{\text{pos}}$ | loss $-\ln p_{\text{pos}}$ |
|---|---|---|---|
| (1.2, −0.3) | 3.320, 0.741 | 0.741 / 4.061 = **0.182** | **1.70** (confident and wrong: big loss) |
| (−1.0, 2.0) | 0.368, 7.389 | 7.389 / 7.757 = **0.953** | **0.049** (confident and right: tiny loss) |

Training pushes the weights so that, on average over the batch, predictions move from the first row towards the second.

**How big is the job?** 2,000 reviews ÷ 16 per batch = **125 steps**. The head adds a 768×768 layer (590,592 weights) and a 768×2 output layer (1,538 weights): 592,130 new parameters on top of the 66 million pre-trained ones. All of them are updated at every step.

## 6. In code

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5)

for texts, labels in batches_of_16:
    batch = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    loss = model(**batch, labels=labels).loss       # forward pass + cross-entropy
    loss.backward(); optimiser.step(); optimiser.zero_grad()
```

The same five-line loop as Module 08. Hugging Face also offers a `Trainer` class that wraps it, with evaluation and checkpointing built in; writing the loop yourself once shows there is no magic inside.

## 7. On real data

- **Before fine-tuning**, the random head scores 41.8% on the 2,000 test reviews: no better than guessing.
- **Training** on 2,000 balanced reviews (125 steps) took **10 to 16 seconds** on the Apple GPU these notebooks were run on; on an ordinary laptop CPU expect a few minutes. The loss falls from **0.694** (the ln 2 of guessing) to **0.189** over the last 10 steps.
- **After fine-tuning: 90.6%.**

| method | our labelled data | test accuracy |
|---|---|---|
| zero-shot (09.5 router model, labels *positive* / *negative*) | none | 81.7% |
| TF-IDF + logistic regression | the same 2,000 reviews | 84.7% |
| ready-made sentiment model (film reviews, 09.5) | none by us | 89.0% |
| **DistilBERT fine-tuned here** | **2,000 reviews** | **90.6%** |

In the exercises, fine-tuning on only **500** reviews still reaches **88.9%**, where TF-IDF reaches 78.0% and embeddings 81.7% with the same 500. Freezing the body and training only the head is faster but drops to 86.4%.

It is not magic. On the tricky sentences, the fine-tuned model gets *"Not bad at all"* right but only just (0.59), and is still fooled by *"I can't fault this cake stand"* (0.18) and by sarcasm (*"Brilliant, the mug arrived in three pieces"*: 0.95). Its most confident "mistakes" on the test set turn out to be mostly **mislabelled reviews**.

The fine-tuned model is saved to `models/shopwave-sentiment-distilbert` (269 MB, git-ignored) for Module 10.

### When to fine-tune, and when to prompt an LLM

| | fine-tune a small model | prompt a large LLM (09.8) |
|---|---|---|
| labelled data | a few hundred or more | none or a handful of examples |
| task | fixed and high-volume (route every email) | changing, varied, or needs free text |
| cost per prediction | tiny, runs on a laptop | API fees or a big local model |
| speed | thousands per second on a GPU | about one per second |
| privacy | data never leaves your machine | depends on provider (or use Ollama) |
| explanations, summaries, replies | no | yes |

ShopWave would fine-tune for routing and sentiment (stable, high volume, labels available) and use an LLM to draft replies (09.8–09.10).

## 8. Where it breaks

- **Too many epochs on too little data overfits.** Watch a validation set, exactly as in 08.6.
- **Too high a learning rate erases pre-trained knowledge** (sometimes called catastrophic forgetting).
- **It inherits the base model's limits**: a 128-token window here, English only, and whatever biases the pre-training text contained.
- **Results vary a little** with the random seed and the hardware; differences under about half a point between runs are noise.
- **A model is something to maintain.** Products, slang and complaint types change; plan to re-label and retrain.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: softmax and cross-entropy loss for logits (1.2, −0.3) and (−1.0, 2.0) when the review is positive.
2. Fine-tune on only 500 reviews. How much accuracy do you lose?
3. Freeze the body and train only the head. Compare time and accuracy.
4. Print the 5 test reviews the fine-tuned model is most confidently wrong about. Model mistakes or label mistakes?

**Quiz** (answers at the end of the notebook)

1. In one sentence, what is fine-tuning?
2. Why does the training loss start at about 0.693?
3. Why is the learning rate so small?
4. When is fine-tuning a small model a better choice than prompting a large LLM?
5. Give two situations where prompting an LLM is the better choice.

## 10. Recap

- **Fine-tuning** = pre-trained body + new head + a short, gentle training run on your labelled data.
- Cross-entropy loss $-\ln p_{\text{true}}$ starts at $\ln 2 = 0.693$ for a guessing head and fell to 0.19 in 125 steps.
- On 2,000 ShopWave-style reviews: **90.6%**, the best in the module; 88.9% from just 500 reviews.
- Fine-tune for stable, high-volume tasks with labels; prompt an LLM for varied tasks, free text and when labels are scarce.

**Next:** [09.7 How large language models work](../07-how-llms-work/): from a 67-million-parameter encoder to billion-parameter models that write.
