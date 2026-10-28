# Prompting, RAG and LLM safety

Learn it in [09.7](../modules/09-nlp-genai/07-how-llms-work/) → [09.12](../modules/09-nlp-genai/12-agents-tool-calling/).

## The course helper: `lib/course_llm.py`

```python
from course_llm import ask, ask_json, chat, info, self_test, last_usage

info()                                          # which provider and model?
self_test()                                     # can this model do what Module 09 needs?
answer = ask("Summarise in one line: ...", system="You are ShopWave's support assistant.")
data = ask_json("Extract product and sentiment: ...", schema, temperature=0)   # -> dict
reply = chat([{"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello! How can I help?"},
              {"role": "user", "content": "Where is my order?"}])
print(last_usage)                               # {"input": tokens, "output": tokens}
```

| Function | Returns | Arguments |
|---|---|---|
| `ask(prompt, system=None, max_tokens=16000, temperature=None)` | text | one question |
| `ask_json(prompt, schema, system=None, max_tokens=16000, temperature=None)` | dict matching `schema` | structured extraction / classification |
| `chat(messages, system=None, max_tokens=16000, temperature=None, schema=None)` | text | multi-turn conversation |
| `info()` / `self_test()` | provider + model / pass-fail check | — |

Choose the provider in `.env` (copy [`.env.example`](../.env.example)): `LLM_PROVIDER=ollama | anthropic | openai | gemini | groq | lmstudio`, optional `LLM_MODEL`, the matching API key, and `LLM_CONTEXT` (Ollama window, default 8192). Which Ollama model for your RAM: [setup/README.md](../setup/README.md#running-a-model-on-your-own-computer-with-ollama). `temperature` is ignored for current Claude models, which do not accept it.

## Prompt patterns

| Pattern | Example | When |
|---|---|---|
| **Zero-shot** | `Classify this email as delivery, refund, product question or complaint.` | Clear task, common categories |
| **Few-shot** | add 2–4 labelled examples before the real input | Your categories have house rules (*20 emails: 80% → 100%*) |
| **Role / system** | `system="You are ShopWave's support assistant. Be brief and polite."` | Tone, persona, standing rules |
| **Delimiters** | `Review: """{text}"""` or `<review>{text}</review>` | Always separate instructions from data |
| **Definitions** | list what each category means | Ambiguous labels (also reached 100% in 09.8) |
| **Step by step** | `Work out the total step by step, then give the final amount.` | Arithmetic, multi-step logic (*£36.25 wrong → £37.70 right*) |
| **Structured output** | `ask_json(prompt, schema)` | Anything a program will read next |
| **Allow "I don't know"** | `If the answer is not in the documents, say "I don't know".` | Factual Q&A, RAG |

A JSON schema for `ask_json`:

```python
schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "complaint": {"type": "boolean"},
    },
    "required": ["product", "sentiment", "complaint"],
    "additionalProperties": False,
}
```

**Prompt checklist:** say the task first · give the input inside delimiters · define every label · show 2–4 examples if labels are subtle · ask for a fixed format (schema) · allow "I don't know" · use `temperature=0` for extraction and classification · test on at least 20 real cases before trusting it.

## RAG checklist

```
documents ─► chunk ─► embed ─► store            (once, and when documents change)
question  ─► embed ─► top-k chunks ─► prompt(instructions + chunks + question) ─► answer + citations
```

| Step | Default in the course | Tune when |
|---|---|---|
| Chunk size / overlap | 80 words / 20 overlap → ⌈(n − o)/(s − o)⌉ chunks | Answers cut mid-sentence (raise overlap); chunks mix topics (smaller size) |
| Embeddings | `all-MiniLM-L6-v2` (local, free) | Domain-specific vocabulary |
| Vector store | Chroma | Millions of chunks |
| Top-k | 3 | *k = 1/3/5 → 13/14/13 of 15 facts; prompt 203/370/544 tokens* |
| Citations | `[returns.md]` after each claim | Always: lets people check the source |
| "I don't know" | when the best similarity is low (*0.23–0.28 for off-topic questions*) | Always test questions the documents do not answer |

→ [09.10](../modules/09-nlp-genai/10-rag/), documents in [data/shopwave_docs](../data/shopwave_docs/)

## Evaluation checklist

1. Write an **eval set** first: 15+ questions with the expected key fact, including some the documents cannot answer.
2. Measure **retrieval** (is the right document in the top-k?) separately from **answers** (does the answer contain the fact?).
3. Automatic checks first (fact present, valid JSON, cites a source); an **LLM-as-judge** with a rubric second. *The judge agreed with the fact check on only 10 of 15.*
4. Re-run the eval after every prompt, model or chunking change. Small local models vary between runs even at temperature 0.
5. Read the failures yourself.

→ [09.11](../modules/09-nlp-genai/11-evaluating-llms/)

## Safety

| Risk | What happens | Mitigation |
|---|---|---|
| **Hallucination** | Confident wrong facts (*"30 days" for sale items; truth: 14*) | RAG with citations; "I don't know"; evals |
| **Prompt injection** | A document or email says "ignore previous instructions…"; the model obeys (*with and without a guard line*) | Treat retrieved text as data; check outputs in code (block card numbers, links); limit what tools can do |
| **Agents doing damage** | Loops, wrong tool, invented arguments (*copied an invoice number from an error message*) | Max steps, validate tool arguments, human approval for refunds and payments |
| **Personal data (PII)** | Names, emails, addresses sent to a third-party API | Send only what is needed; mask or use a local model; check the provider's data policy |
| **API keys** | Key committed to git or pasted in a notebook | `.env` only (git-ignored); never print it; rotate a leaked key immediately |
| **Bias** | Different answers for different names or groups | Test with swapped names; review samples by group |

## Cost

$$\text{cost} = \frac{\text{input tokens}}{10^6}\times\text{input price} + \frac{\text{output tokens}}{10^6}\times\text{output price}$$

Worked example (09.9): 1,000 reviews × (67 input + 6 output tokens) on a model priced $0.10 / $0.50 per million tokens:

$$\frac{67{,}000}{10^6}\times 0.10 + \frac{6{,}000}{10^6}\times 0.50 = 0.0067 + 0.0030 \approx \$0.01$$

The same job on a $4 / $20 model costs about **$0.39**. A 400-token system prompt sent with every request multiplies input tokens about 7×: count the instructions, not just the data. Prices change; check the provider's page. Local models (Ollama) cost nothing per token but are smaller and slower. → [09.9](../modules/09-nlp-genai/09-calling-llms-from-python/)
