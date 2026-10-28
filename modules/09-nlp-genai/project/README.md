# 09.P · Project: review analyser and support chatbot

⏱ about 2 hours · Prerequisites: all of Module 09 · Notebooks: [project.ipynb](project.ipynb) (your work), [solution.ipynb](solution.ipynb) (one complete answer)

---

## The brief

ShopWave's head of customer experience has two requests.

> *"Every month someone spends a day reading product reviews to tell the product team what is going wrong. Can a model do the counting? And can we have an assistant that answers the policy questions that fill half our support inbox? Before anything goes near a customer, I need to know how often it is right, what it costs, and how it fails."*

You will build both and write the report she asked for.

**Part A · Review analyser.** For each of 30 hand-labelled reviews, extract the product, the sentiment (positive / neutral / negative) and the main issue (none, damaged, late_delivery, wrong_item, quality, price). Measure accuracy against the labels, study the mistakes, improve the prompt once, and produce a chart of the problems customers mention.

**Part B · Support chatbot.** Build the RAG assistant over `data/shopwave_docs/` (09.10): cited answers, *I don't know* when the documents do not cover a question. Evaluate it on the 15-question set from 09.11: fact accuracy, retrieval hit@3, and the failures.

**Part C · Write-up.** Measure tokens and seconds per review and per answer, estimate the cost of 10,000 reviews on two hosted models, and write: what works, a table of failures each with a fix, cost and speed, and a one-paragraph recommendation.

All LLM calls go through `lib/course_llm.py`, so the project runs with whatever provider you set up (a free local model is fine).

## What you hand in

- `project.ipynb` running top to bottom without errors.
- Two accuracy numbers for Part A, before and after your prompt improvement, and the chart.
- Fact accuracy and hit@3 for Part B, with the failures shown.
- The write-up as a markdown cell at the end.

## Rubric (20 points)

| Area | Points | Full marks when… |
|---|---|---|
| Extraction design | 4 | System prompt defines every field; JSON schema with `enum`s; temperature 0; review delimited |
| Measuring Part A | 4 | Accuracy for both fields, a sentiment confusion table, mistakes listed and explained, one measured prompt improvement |
| RAG pipeline | 4 | Chunk, embed, store, retrieve, generate with citations and *I don't know*; runs end to end |
| Evaluating Part B | 4 | Fact accuracy and hit@3 reported; each failure diagnosed as retrieval or generation (or a too-strict check) |
| Write-up | 4 | Numbers come from your own run; failures table with concrete fixes; honest recommendation that matches the evidence |

**Stretch goals**

- Run both parts with a second model (a larger Ollama model or an API) and compare accuracy, speed and cost in one table.
- Give the chatbot memory: send the last two turns, and rewrite the question before retrieval so "it" is resolved.
- Use an LLM-as-judge (09.11) on the chatbot answers and measure its agreement with your own grades.
- Run the review analyser on a few hundred real Amazon reviews with `load_reviews()` from `data/download.py`, using sentiment only (those reviews have no issue labels).

## One solution's results

[solution.ipynb](solution.ipynb), run with `llama3.2` (3B) via Ollama:

- **Review analyser:** sentiment **83%** and issue **80%** correct on 30 reviews. Every negative and positive review got the right sentiment; the neutral ones were the problem (2 of 7 right). Issue mistakes confused similar labels: a crushed parcel tagged *wrong_item* instead of *damaged*.
- **Chatbot:** key fact in **14/15** answers; right document in the top 3 for **13/13** answerable questions; *I don't know* for both uncovered questions. The one "failure" was a correct answer worded differently from the check.
- **Cost and speed:** about 119 input and 27 output tokens per review, so 10,000 reviews cost about **$0.25** on Claude Haiku 5.5 or **$10** on Claude Opus 5.5 (October 2026 prices), and about a second or less per review on a laptop.
- **Recommendation:** use the analyser for monthly reporting now; use the chatbot to draft replies for the support team (a person sends them) until it passes a larger eval set with a stronger model.

Your numbers will differ with a different model or prompt. What matters is that every claim in your write-up comes from a number you measured.
