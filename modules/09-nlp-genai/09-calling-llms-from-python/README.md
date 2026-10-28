# 09.9 · Calling LLMs from Python

⏱ about 1.5 hours · Prerequisites: 09.8 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The prompts from 09.8 work in a notebook. Now ShopWave's head of operations asks the questions every real project meets:

- *"Which provider? Can we switch later without rewriting everything?"*
- *"What will it cost at 10,000 emails a day?"*
- *"How fast is it? What happens when the service is down?"*
- *"Where does the API key live, and who can see it?"*

So far you have used `ask()` without looking inside. This lesson opens it: the raw calls each provider expects, why the course wraps them in one helper, and the arithmetic of tokens, cost, latency and failure.

## 2. The idea

Every LLM provider sells the same thing: **you send messages, you get text back, you pay per token.** It is like posting a letter: the envelope format differs between countries (where the stamp goes, how the address is written), but the letter inside is the same.

Each provider's API is one envelope format. There are really only two in common use: Anthropic's own (the Anthropic SDK) and the OpenAI chat format, which most others (Gemini, Groq, LM Studio, Ollama) also accept. Ollama has its own native format too. A small **adapter** that knows the formats lets the rest of your code ignore which envelope is in use, and that is what `lib/course_llm.py` is.

## 3. Definition

- **API (application programming interface)**: a web address you send a request to and get structured data back from.
- **SDK (software development kit)**: a library that wraps an API in normal functions, such as `anthropic.Anthropic().messages.create(...)`.
- **API key**: a secret string that identifies you to the provider and is billed for your usage. Whoever holds it can spend your money.
- **Input tokens / output tokens**: what you send and what the model writes. Priced separately, output usually several times higher.
- **Latency**: time from sending a request to having the answer.
- **Rate limit**: the maximum requests or tokens per minute the provider allows; going over returns an error (HTTP 429).
- **Exponential backoff**: retry a failed request after 1 s, then 2 s, then 4 s, and so on, so you do not hammer a struggling service.

## 4. Formula

$$\text{cost} = N \times \frac{t_{in} \times p_{in} + t_{out} \times p_{out}}{1{,}000{,}000}$$

- $N$: number of requests
- $t_{in}$, $t_{out}$: average input and output tokens per request
- $p_{in}$, $p_{out}$: price per **million** tokens, input and output

Prices change often; always check the provider's pricing page. In this lesson (Claude, October 2026): Haiku 5.5 $0.10 / $0.50, Sonnet 5.5 $2 / $10, Opus 5.5 $4 / $20 per million input / output tokens. A local Ollama model has no per-token price; you pay in hardware and electricity.

## 5. By hand

ShopWave wants sentiment and product name for **1,000 reviews**. Suppose each request is 150 input tokens (instruction + review) and each answer 40 output tokens.

**Claude Haiku 5.5** ($0.10 in, $0.50 out):
input 1,000 × 150 = 150,000 tokens → 0.15 × $0.10 = $0.015;
output 1,000 × 40 = 40,000 tokens → 0.04 × $0.50 = $0.020;
**total $0.035.**

**Claude Opus 5.5** ($4 in, $20 out): 0.15 × $4 + 0.04 × $20 = $0.60 + $0.80 = **$1.40.**

Now add the 400-token few-shot system prompt from 09.8 to every request. Input becomes 550 tokens per request:
Haiku: 0.55 × $0.10 + 0.04 × $0.50 = $0.055 + $0.020 = **$0.075** (more than double);
Opus: 0.55 × $4 + 0.04 × $20 = $2.20 + $0.80 = **$3.00.**

The extra accuracy from few-shot examples is real, and so is the cost: you pay for the examples on every single request.

## 6. In code

[lesson.ipynb](lesson.ipynb):

1. **Three raw calls, same request.** The Anthropic SDK (`client.messages.create(model=..., max_tokens=..., system=..., messages=[...])`, which runs only if you have `ANTHROPIC_API_KEY`); the OpenAI client pointed at Ollama's `/v1` address; and Ollama's native `/api/chat` with plain JSON. Each prints the answer and the token counts.
2. **The course helper.** `inspect.getsource` prints how it chooses a provider and its whole Claude path, so you can see it is the same SDK call plus structured output and a refusal fallback.
3. **Tokens and cost, measured.** Five real reviews averaged 67 input and 6 output tokens with `llama3.2` (much less than the 150/40 assumed above, because the answers were very short). At those sizes 1,000 reviews cost about $0.01 on Haiku 5.5 and $0.38 on Opus 5.5; adding a 400-token system prompt raises them to $0.05 and $1.98.
4. **Latency.** About 0.12 s per review on a laptop with the model already loaded, so roughly 2 minutes for 1,000 one after another.
5. **Retries.** A retry-with-backoff wrapper, demonstrated against a model name that does not exist: it waits 1 s, then 2 s, then gives up with a clear message.

> Outputs in the notebook were produced with `llama3.2` (3B) via Ollama; yours will differ.

## 7. On real data

At ShopWave's volume (10,000 support emails a day, exercise 1) the choice of model is the biggest cost lever: the same classification costs about **$17 a month** on Haiku 5.5 and **$684 a month** on Opus 5.5. The engineering habit is to start with the cheapest model that passes your test set (09.8's measured accuracy), and only move up when the numbers say you must.

## 8. Where it breaks

- **Leaked keys.** A key pasted into a notebook ends up in git history, a screenshot or a shared file. Keep keys in `.env` (git-ignored), revoke immediately if one leaks, and set a spending limit per key.
- **Silent cost growth.** Long system prompts, chat histories that are re-sent every turn, and retry loops multiply input tokens. Log tokens per call (exercise 2).
- **Assuming the reply shape.** Current Claude models can return thinking blocks before the text; `response.content[0].text` is fragile. Join the text blocks.
- **Retrying the unfixable.** Retry timeouts, rate limits and server errors; fail fast on a wrong key, a wrong model name or a bad request.
- **Provider lock-in.** Calling one SDK directly everywhere makes switching expensive. Keep provider calls behind one small function, as the course helper does (LiteLLM does this for 100+ providers in production).
- **Different models, different numbers.** Token counts, speed and answers all change when you switch model. Re-measure.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Cost per day and per month for 10,000 emails with a 400-token system prompt, on Haiku 5.5 and Opus 5.5: by hand, then in code.
2. Write `logged_ask` that records characters, tokens and seconds for every call. How does input length affect latency?
3. Switch provider or model in `.env`, run `self_test()`, and compare answers, speed and tokens.
4. Find every problem in a short code snippet with a hard-coded key, and rewrite it.

**Quiz**

1. Which two kinds of token do you pay for, and which is usually more expensive?
2. Cost of 1,000 requests of 150 input and 40 output tokens on a model priced $0.10 / $0.50 per million?
3. Why does a long system prompt matter so much for cost?
4. When you switch between OpenAI-compatible providers, what changes in the code?
5. Which errors are worth retrying?

## 10. Recap

- Every provider does the same thing: messages in, text out, billed per input and output token.
- Two common request formats (Anthropic's and the OpenAI chat format) plus Ollama's native API cover every provider in this course; `course_llm.py` hides the differences.
- **Cost = requests × (input tokens × input price + output tokens × output price) / 1,000,000.** System prompts are paid for on every request.
- Keep keys in `.env`, never in code; retry only errors that can heal, with exponential backoff.
- Choose the cheapest model that passes your measured test, not the most famous one.

**Next:** [09.10 Retrieval-augmented generation](../10-rag/) gives the model ShopWave's own policies to answer from.
