# 09.12 · Agents and tool calling

⏱ about 1.5 hours · Prerequisites: 09.8 (structured output), 09.11 (prompt injection), 01.6 (Pandas lookups) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The RAG assistant (09.10) answers questions about **policies**. But many customer questions are about **their own data**: *"Where is order 536365 and how much was it?"*, *"Do you sell hot water bottles, and how much are they?"*. Those answers live in ShopWave's order and product tables, not in any document, and they change every minute.

You could write a separate program for each kind of question. Or you can give the model a few **tools** (Python functions that look things up) and let it decide which to use for each question. That is an **agent**. It is the most hyped idea in AI right now, and this lesson shows both why it is useful and why it needs guardrails.

## 2. The idea

A new support agent on day one has access to three internal systems: the order lookup screen, the product catalogue, and the policy handbook. For each customer email they decide which system to check, read what it says, maybe check another, then reply.

An LLM agent does the same in a loop:

```
question ─► model decides: which tool, with what argument? ─► our code runs the tool
              ▲                                                        │
              └──────────── model sees the result ◄────────────────────┘
          … until the model decides to give a final answer (or runs out of steps)
```

Crucially, **the model never runs anything itself.** It only asks; our code checks the request and runs it. Everything that can go wrong (and in this lesson a lot does) can be caught at that point.

## 3. Definition

- **Tool** (function): a Python function the model may ask to call, described by a name, a purpose and its argument.
- **Tool calling** (function calling): the model replies with a structured request, such as `{"tool": "order_status", "argument": "536365"}`, instead of (or before) a text answer.
- **Agent**: a loop in which a model repeatedly chooses tool calls, sees their results and decides what to do next, until it gives a final answer.
- **Trace**: the log of every step: decision, tool, argument, result.
- **Guardrail**: a check that limits what the agent can do: in the prompt (rules) or, more reliably, in code (validation, step limits, budgets, approval queues).
- **Least privilege**: give an agent only the tools it needs, read-only by default.

We implement tool calling with `ask_json`-style structured output that works with every provider. Providers also offer **native** tool-calling APIs (Anthropic tool use, OpenAI function calling) that do the same thing with built-in support; the loop is identical.

## 4. Formula

No new mathematics, but one cost formula worth knowing. Each step re-sends the whole conversation so far, so the input tokens of step $s$ are roughly

$$t_{in}(s) \approx t_{system} + t_{question} + \sum_{j < s} \big(t_{decision,j} + t_{result,j}\big)$$

and the total grows roughly with the **square** of the number of steps. A confused agent that loops is not just slow; it is increasingly expensive. Hence step limits and token budgets.

## 5. By hand

Be the agent. Three tools:

| Tool | Argument | Returns |
|---|---|---|
| `order_status` | 6-digit invoice | date, lines, total, country, status |
| `refund_policy` | none | the returns and refunds rules |
| `product_search` | product words | up to 5 matching products with typical price |

For each customer message, write down the sequence of tool calls you would make, and your final answer:

1. *"What's the status of order 536365, and how much was it?"* → `order_status("536365")` → final answer with date and total. Two steps.
2. *"Do you sell hot water bottles? Roughly how much?"* → `product_search("hot water bottle")` → answer with the price range.
3. *"I bought a cushion three weeks ago and don't like it. Can I get a refund?"* → no invoice number given! The right move is a final answer **asking for the invoice number** (or explaining the 30-day rule from `refund_policy`), not inventing a number.
4. *"Order 536365: can I still return it today?"* → `order_status` gives 1 Dec 2010. 30 days later is 31 Dec 2010, long past, so no. That last step is **date arithmetic**, which you did in your head and the model may not.

## 6. In code

[lesson.ipynb](lesson.ipynb) runs the four questions twice.

**A naive agent** (tools + instructions + loop). In our run with `llama3.2` it:

- answered question 1 correctly but called `refund_policy` twice for no reason and replied in **JSON** instead of a sentence;
- called `product_search` three times with the same argument before answering question 2;
- on question 3, called `order_status("1234567890")`, got an error message that said *"for example 536365"*, and then **used 536365 as the customer's order**. Text in a tool result steers the model;
- on question 4, concluded "no returns accepted after delivery", which is the right outcome for the wrong reason.

**A guarded agent**: tool errors that instruct instead of giving examples, prompt rules (no repeats, ask for missing information, plain sentences) and a code check that refuses to re-run an identical call. Same model, same questions:

- question 1: two steps, one clear sentence with date and total;
- question 3: **asks the customer for their invoice number**;
- question 2 still took five steps, because "hot water bottles" (plural) matches nothing; question 4 still gets the reason wrong.

The exercises fix those last two **in the tools**: a plural-tolerant search, and a `return_deadline` tool that does the date arithmetic in Python.

> Outputs in the notebook were produced with `llama3.2` (3B) via Ollama; yours will differ. Larger models make fewer of these mistakes, but the same kinds.

## 7. On real data

The tools run on ShopWave's real order lines (Online Retail II): invoice 536365 really is a £139.12 order from 1 December 2010, and the hot water bottles really cost £4.25–£4.95. Production agents look the same, with more tools (order changes, delivery tracking, CRM notes) and much stricter controls: authentication (a customer may only look up *their own* orders), audit logs of every step, and human approval for any action that moves money.

## 8. Where it breaks

- **Invented arguments.** Missing information gets filled in with plausible guesses. Validate every argument in code and tell the model to ask instead.
- **Tool results are prompts.** An example number in an error message, or an injected instruction in a retrieved page (09.11), steers the next decision.
- **Loops.** Repeating calls or trying endless variations. Cap steps and tokens; refuse identical repeats in code.
- **Reasoning on facts.** Dates, sums and eligibility rules are where small models slip. Put them in tools.
- **Too much power.** An agent with a `issue_refund` tool and a prompt injection is a security incident. Read-only by default; actions go to an approval queue (exercise 4).
- **Hard to test.** The path through the tools varies with wording. Evaluate final answers *and* traces on a fixed question set, as in 09.11.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Make `product_search` tolerate plurals and empty queries; re-run the hot water bottle question.
2. Add a `return_deadline(invoice)` tool that computes the 30-day deadline in Python, and test it on orders 581587 and 536365.
3. Add a token budget of 3,000 per conversation. Which questions come closest?
4. Write an `issue_refund` tool that only queues requests for a person. Which other actions should work this way?

**Quiz**

1. What is an agent, in one sentence?
2. Who actually runs the tool: the model or your code?
3. Why did the naive agent use invoice 536365 for a customer who never gave a number?
4. What is a step limit for?
5. Which kinds of action should never be executed by an agent without a check?

## 10. Recap

- An agent is a loop: the model **chooses a tool**, your code **runs it**, the model **reads the result**, until it answers.
- Tool calling can be done with structured output (as here) or with a provider's native tool API; the loop is the same.
- Small models invent arguments, repeat calls, answer in the wrong format and reason badly about dates; you saw all four.
- The best guardrails are **in code**: argument validation, refusing repeats, step and token limits, read-only tools, approval queues.
- Let the model handle language and decisions; let code handle facts, arithmetic and permissions.

**Next:** [09.P Project: review analyser and support chatbot](../project/) puts structured extraction, RAG and evaluation together.
