# 09.7 · How large language models work

⏱ about 1.5 hours · Prerequisites: 09.3 (embeddings), 09.4 (attention and Transformers), 08.3 (softmax) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's support inbox gets 400 emails a day. Someone in the leadership meeting says: *"Just put ChatGPT on it."* Someone else says: *"It makes things up. Last week it told my friend a shop had a 90-day return policy that doesn't exist."*

Both are partly right, and you cannot judge who until you know what a large language model actually does. Is it looking things up? Reasoning? Remembering? This lesson opens the box. Everything in the rest of the module (prompting, RAG, evaluation, agents) is a way of working *with* the mechanism you meet here, or *around* its weak spots.

## 2. The idea

Your phone keyboard suggests the next word as you type. Type *"see you"* and it offers *"tomorrow"*, *"soon"*, *"later"*. It learned that from what you and millions of others typed before.

A large language model is that keyboard, scaled up enormously. It has read a large part of the public internet, books and code, and it has learned, for any piece of text, how likely each possible next piece is. To write an answer it picks a next piece, adds it to the text, and asks itself again. One piece at a time, until it decides to stop.

That explains both sides of the meeting argument. The model is fluent and often right, because the most likely continuation of a sensible question is usually a sensible answer. But it is **not looking anything up**. If it has never seen ShopWave's return policy, the most likely continuation is a *typical* return policy, stated just as confidently.

## 3. Definition

- **Language model**: a model that assigns a probability to every possible next token, given the text so far.
- **Token**: the unit a model reads and writes: a common word, a piece of a rarer word, a digit group or punctuation. In English, roughly 3–4 characters on average.
- **Large language model (LLM)**: a Transformer (09.4) with billions of parameters, trained on a huge amount of text to predict the next token.
- **Decoding**: turning those probabilities into text. **Greedy** decoding always takes the most likely token; **sampling** draws a token at random according to the probabilities.
- **Temperature**: a number that sharpens (low) or flattens (high) the probabilities before sampling.
- **Top-p (nucleus sampling)**: sample only from the smallest set of top candidates whose probabilities add up to *p*.
- **Context window**: the maximum number of tokens the model can take into account at once: your instructions, any documents, the conversation and its own answer.

How models are made, in three stages:

1. **Pre-training**: predict the next token over trillions of tokens of text. The model learns language, facts and patterns, but it is not yet a helpful assistant: ask it a question and it may continue with three more questions.
2. **Instruction tuning**: further training on many examples of requests and good answers. Now it answers instead of continuing.
3. **Preference training** (for example **RLHF**, reinforcement learning from human feedback): people compare pairs of answers; the model is trained to produce the kind they prefer: helpful, honest, safe.

**Open-weight vs proprietary models.** Open-weight models (Llama, Qwen, Gemma, Mistral, gpt-oss) can be downloaded and run on your own computer; you pay with your hardware and they are usually smaller. Proprietary models (Claude, GPT, Gemini) run on the provider's servers; you pay per token and they are usually larger and stronger. This course works with both through one helper (09.9).

## 4. Formula

A bigram model estimates the probability of the next word from counts:

$$P(w_{next} \mid w) = \frac{\text{count}(w,\ w_{next})}{\text{count}(w,\ \text{anything})}$$

An LLM instead produces a raw score $z_i$ (a **logit**) for every token $i$ in its vocabulary. Softmax with temperature $T$ turns scores into probabilities:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T = 1$: the model's own probabilities.
- $T < 1$: differences between scores are magnified; the favourite wins more often.
- $T > 1$: differences shrink; unlikely tokens get a real chance.
- $T \to 0$: always the top token (greedy).

## 5. By hand

**A bigram model on three sentences.** *the mug is red · the mug is big · the lamp is red*

| after | next word counts | probabilities |
|---|---|---|
| *the* | mug 2, lamp 1 | mug 2/3, lamp 1/3 |
| *mug* | is 2 | is 1 |
| *lamp* | is 1 | is 1 |
| *is* | red 2, big 1 | red 2/3, big 1/3 |

Greedy decoding from *the*: *mug* (2/3) → *is* → *red* (2/3). Output: **the mug is red**. You have built a language model.

**Temperature on four candidates.** The text so far is *"Thank you for your …"* and suppose the model scores four candidates: *order* 3.0, *purchase* 2.5, *patience* 1.5, *banana* −1.0.

At $T = 1$: $e^{3.0} = 20.086$, $e^{2.5} = 12.182$, $e^{1.5} = 4.482$, $e^{-1.0} = 0.368$; sum = 37.118.

| | order | purchase | patience | banana |
|---|---|---|---|---|
| $T = 0.5$ (scores 6, 5, 3, −2; sum of exps 572.06) | **0.705** | 0.259 | 0.035 | 0.000 |
| $T = 1$ (20.086 / 37.118 …) | **0.541** | 0.328 | 0.121 | 0.010 |
| $T = 2$ (scores 1.5, 1.25, 0.75, −0.5; sum 10.696) | **0.419** | 0.326 | 0.198 | 0.057 |

At $T = 0.5$ the favourite takes 70.5%; at $T = 2$ even *banana* gets 5.7%.

**Top-p = 0.9 at $T = 1$.** Sort and add up: order 0.541 → + purchase 0.869 → + patience 0.990. The total passes 0.9 at *patience*, so sampling is restricted to {order, purchase, patience}. *Banana* can never be picked, however high the temperature.

## 6. In code

[lesson.ipynb](lesson.ipynb) rebuilds both tables, then:

- counts tokens with a real tokenizer: *"Thank you for your order."* is 6 tokens, *"ShopWave's WELCOME10 code gives 10% off."* is 13 (`WELCOME10` alone is three), and a five-word Indonesian sentence is 9;
- asks Ollama for the probabilities `llama3.2` gave its top five candidates after *"Thank you for your"*. The favourite, `help`, gets only 18.6%, and one candidate, `gener`, is the first piece of *generosity*;
- asks for a candle name five times at temperature 0 (one answer, five times) and at 1.2 (four different names).

## 7. On real data

All eleven of ShopWave's policy documents together come to about 1,750 tokens. They would fit in the course's default 8,192-token context window. Two years of transactions (half a million rows) would not fit in any model's window. That is why lesson 09.10 retrieves only the relevant paragraphs instead of pasting everything in.

> Outputs in the notebook were produced with `llama3.2` (3B) via Ollama; yours will differ.

## 8. Where it breaks

- **Fluent is not true.** The model writes the most plausible continuation. Without the real document in front of it, a plausible return policy is an invented one (09.11 measures this).
- **No memory between calls.** Each request starts from nothing. A chatbot "remembers" only because your code sends the whole conversation again every time, which costs tokens.
- **Knowledge has a date.** Pre-training stops at a cut-off; anything later, or anything private (like ShopWave's policies), the model has never seen.
- **Tokens are not words.** Counting letters, reversing strings and arithmetic on long numbers are hard, because the model sees `WELCOME10` as three chunks, not nine letters and digits.
- **Temperature is not a quality dial.** Low temperature makes answers repeatable, not correct. Current Claude models do not accept it at all; the course helper ignores it for them.
- **Context is finite.** Long conversations and big documents get cut, or cost a lot; models also use the middle of a very long context less reliably than the start and end.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Add *"the lamp is big"* and *"the mug is blue"* to the bigram model. By hand: P(*mug* | *the*), P(*red* | *is*) and the greedy output. Then check in code.
2. Compute the softmax of the four scores at T = 0.1 and T = 5. What happens as T → 0 and T → ∞?
3. Count words and tokens for a product description, its translation into another language, a 16-digit order number and an email address. Which is most expensive per character?
4. Ask a one-word sentiment question five times at temperature 0 and at 1.2. Does temperature change the answer? Why or why not?

**Quiz**

1. What exactly does a language model predict?
2. In the three-sentence model, what is P(*red* | *is*)?
3. What does a model read: words, characters or tokens? Why does it matter for cost?
4. What does lowering the temperature do to the probabilities?
5. Name the three training stages of a modern chat model, one line each.

## 10. Recap

- An LLM repeatedly predicts the next **token** from everything before it; text is generated one token at a time.
- It is a much larger version of the bigram table you built by hand, with context of thousands of tokens.
- Softmax turns scores into probabilities; **temperature** sharpens or flattens them; **top-p** cuts off the unlikely tail.
- Pre-training gives knowledge and fluency; instruction tuning and preference training make it a helpful assistant.
- It does not look things up: private or recent facts must be put into the prompt. That is what the rest of this module does.

**Next:** [09.8 Prompt engineering](../08-prompt-engineering/) shows how much the wording of a request changes the answer, and measures it.
