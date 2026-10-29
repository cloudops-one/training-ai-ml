# From Python to AI: Solve Real Problems with Machine Learning

A free, self-paced course that takes you from "I can write Python" to "I can build, explain and ship machine-learning and AI systems".

No maths background needed. No ML background needed. Every idea is explained with a real problem first, then a picture, then a formula, then a calculation you do by hand on a handful of rows, then code.

---

## Who this is for

You can write Python: variables, lists, dictionaries, loops, functions, maybe a class or two. That is enough.

You do **not** need statistics, calculus or linear algebra. The small amount of maths that matters is taught exactly where it is needed, with numbers you can check on paper.

## What you will be able to do at the end

- Look at a business problem and decide whether machine learning is the right tool, and which kind.
- Take messy data, clean it, explore it and turn it into features a model can learn from.
- Train, evaluate, tune and explain regression, classification, clustering and forecasting models.
- Build neural networks for images and text, and use transfer learning instead of starting from zero.
- Use large language models from Python: prompting, structured output, retrieval-augmented generation (RAG), simple agents.
- Deploy a model as an API or web app, track experiments, and keep a model healthy in production.
- Present a complete portfolio project and talk about it confidently in an interview.

## How the course works: one story, twelve modules

You have just been hired as the first data person at **ShopWave**, an online store. Every module starts with a problem ShopWave has. You learn exactly what is needed to solve it, nothing more, nothing earlier.

| # | Module | ShopWave asks… | You learn | Hours |
|---|--------|----------------|-----------|-------|
| 00 | [The Big Picture](modules/00-big-picture/) | "Can ML even help us?" | What AI / ML / deep learning / generative AI are, and the 7-step ML workflow | 4 |
| 01 | [Working with Data](modules/01-working-with-data/) | "Make sense of our sales data" | NumPy, Pandas, plotting, exploratory analysis, basic statistics | 20 |
| 02 | [Regression](modules/02-regression/) | "How much will a customer spend?" | Linear regression from scratch, error metrics, gradient descent, tree regressors | 14 |
| 03 | [Classification](modules/03-classification/) | "Which customers will leave?" | KNN, decision trees, logistic regression, confusion matrix, precision/recall, ROC | 16 |
| 04 | [Making Models Trustworthy](modules/04-trustworthy-models/) | "It worked in the notebook, why not live?" | Splits, leakage, scaling, encoding, cross-validation, pipelines, overfitting | 14 |
| 05 | [Powerful Models & Explainability](modules/05-powerful-models/) | "Make it accurate, then explain it to the CEO" | Random forests, gradient boosting, tuning, imbalanced data, SHAP, AutoML | 16 |
| 06 | [Patterns Without Labels](modules/06-unsupervised/) | "Group our customers, find fraud, what sells together?" | Clustering, PCA, anomaly detection, market basket, recommenders | 14 |
| 07 | [Forecasting](modules/07-forecasting/) | "How much stock for December?" | Trend, seasonality, ARIMA, ML forecasting, backtesting | 10 |
| 08 | [Deep Learning](modules/08-deep-learning/) | "Auto-tag our product photos" | Neurons, backpropagation, Keras, CNNs, transfer learning | 24 |
| 09 | [Language & Generative AI](modules/09-nlp-genai/) | "Read 50,000 reviews, answer support chats" | TF-IDF, embeddings, Transformers, LLM APIs, prompting, RAG, agents | 24 |
| 10 | [Shipping & Running ML](modules/10-deployment-mlops/) | "Put it in production and keep it working" | FastAPI, Streamlit, Docker, MLflow, drift, responsible AI | 10 |
| 11 | [Capstone & Next Steps](modules/11-capstone/) | "Show us what you can do" | A full portfolio project, interview prep, where to go next | 20 |

Total: about **190 hours**. At 4 hours a day that is roughly 8 weeks; at 1 hour a day, about 6 months. There are no deadlines. The [ROADMAP](ROADMAP.md) lists every lesson with a checkbox so you always know where you are.

## How each lesson is built

Every lesson follows the same ten steps, so you always know what is coming:

1. **The problem**: what ShopWave needs and why the tools you already have are not enough.
2. **The idea**: an everyday analogy and one picture.
3. **Definition**: precise, with every new word explained the moment it appears.
4. **Formula**: shown in full, every symbol named.
5. **By hand**: worked through on 5–10 rows with real numbers you can check.
6. **In code**: the same numbers in plain Python/NumPy, then the library call. The outputs match.
7. **On real data**: applied to ShopWave.
8. **Where it breaks**: limitations and the mistakes everyone makes once.
9. **Check yourself**: exercises and a short quiz, answers included.
10. **Recap**: five bullets and a link to the next lesson.

Each lesson folder contains `README.md` (read this), `lesson.ipynb` (run this), `exercises.ipynb` (do this) and `solutions.ipynb` (check this).

## Getting started

1. Read [setup/README.md](setup/README.md). You can run everything in Google Colab (nothing to install) or locally.
2. Run `data/download.py` once to fetch the datasets, or let the notebooks fetch them on first use.
3. Start at [Module 00](modules/00-big-picture/). Do not skip it, even if you are impatient: it shows you the whole course in one notebook so you know where you are going.
4. Tick lessons off in [ROADMAP.md](ROADMAP.md).

### Optional: read the course as a website

The lessons are also published as a website with navigation and search: **https://cloudops-one.github.io/training-ai-ml/**. To view it on your own computer instead, run this from the repository root (with your virtual environment active):

```bash
pip install -r website/requirements.txt   # once
python website/build.py serve
```

Then open **http://127.0.0.1:8000/training-ai-ml/** in your browser. Press `Ctrl+C` to stop it. You still run the notebooks in Jupyter or Colab: the website links to them. More detail is in [website/README.md](website/README.md).

## Rules for studying

- **Do the by-hand calculations.** They are short on purpose. This is where understanding happens.
- **Type the code, do not paste it.** At least the first time.
- **Do the exercises before looking at the solutions.** Being stuck for ten minutes is part of learning.
- **Finish each module's mini-project before moving on.** They build on each other.
- **When a word is unfamiliar, look it up in the [GLOSSARY](GLOSSARY.md).**

## Reference material

- [GLOSSARY.md](GLOSSARY.md): every term in one line.
- [cheatsheets/](cheatsheets/): which model when, metrics, Pandas, scikit-learn, Keras, prompting.
- [REFERENCES.md](REFERENCES.md): the books, courses and documentation this course draws on.

## Licence

Content is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and code under the MIT licence. Use it, share it, teach with it.
