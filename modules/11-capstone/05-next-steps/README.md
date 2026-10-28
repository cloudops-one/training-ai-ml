# 11.5 · What to learn next

⏱ about 1 hour · Prerequisites: the rest of the course

---

## 1. Where you are now

You started this course able to write Python. You can now take a vague question, frame it, find and clean data, build and honestly evaluate models from linear regression to transformers, explain them, ship them and watch them. That is the core of the job.

Nobody knows all of machine learning. People go deep in one direction. This page maps the main directions so you can choose one deliberately, rather than drifting into whatever tutorial appears next.

## 2. How to choose

Ask which part of the course you enjoyed most, not which sounds most impressive:

| If you enjoyed… | Look at… |
|---|---|
| Module 04 (pipelines, leakage) and Module 10 (deployment, monitoring) | MLOps and ML platform engineering |
| Module 01 (cleaning, joining, building tables) | Data engineering |
| Module 09 (embeddings, RAG, agents) | NLP and LLM engineering |
| Module 08 (CNNs, transfer learning) | Computer vision |
| Module 06 (segments, market basket, recommenders) | Recommender systems |
| Module 07 (forecasting, backtesting) | Time series and forecasting |
| The by-hand calculations, and wanting to know *why* | The research and maths path |

## 3. The specialisation map

### MLOps and ML platform

**The job:** making models reliable in production: training pipelines, deployment, monitoring, retraining, cost. Closer to software engineering than to statistics.
**Learn next:** containers and orchestration, CI/CD for models, feature stores, model registries, monitoring and alerting.
**Free resources:**
- Chip Huyen, *Designing Machine Learning Systems* (the design chapters; see [REFERENCES](../../../REFERENCES.md))
- [Made With ML](https://madewithml.com) by Goku Mohandas
- [MLflow documentation](https://mlflow.org/docs/latest/) and the [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
**First project:** turn your capstone into a scheduled retraining pipeline that logs every run to MLflow, compares the new model with the current one on a fixed holdout, and only promotes it if it wins.

### Data engineering

**The job:** building the pipelines and tables everyone else's models depend on. In many teams this is where most of the value (and the jobs) are.
**Learn next:** SQL in depth, a cloud data warehouse, orchestration (Airflow, Dagster or Prefect), data quality testing, dbt.
**Free resources:**
- [DataTalks.Club Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- [Mode SQL tutorial](https://mode.com/sql-tutorial)
- [dbt Learn](https://learn.getdbt.com)
**First project:** rebuild the ShopWave customer table from Module 01 as a daily pipeline: raw transactions into a database, SQL transformations, tests that fail on bad data, a scheduled run.

### NLP and LLM engineering

**The job:** building products on language models: retrieval, agents, evaluation, cost and safety. The fastest-moving area; evaluation skill is what separates good engineers.
**Learn next:** advanced retrieval (hybrid search, re-ranking), structured outputs and tool use with provider APIs, evaluation at scale, fine-tuning small open models.
**Free resources:**
- [Hugging Face courses](https://huggingface.co/learn) (NLP and LLM courses)
- [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses) on RAG, agents and evaluation
- Provider documentation, read properly: [Anthropic](https://docs.anthropic.com), [Ollama](https://ollama.com)
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer)
**First project:** extend the 09.P support chatbot with hybrid search (keywords plus embeddings), a 50-question evaluation set, and a report comparing two models on quality, latency and cost.

### Computer vision

**The job:** models that see: product tagging, quality inspection, document reading, medical imaging.
**Learn next:** object detection and segmentation, modern pretrained backbones, data labelling workflows, deploying small models to phones or edge devices.
**Free resources:**
- [fast.ai *Practical Deep Learning for Coders*](https://course.fast.ai)
- [PyTorch tutorials](https://pytorch.org/tutorials)
- Stanford [CS231n notes](https://cs231n.github.io)
**First project:** extend the 08.P product classifier into an object detector that finds and labels several products in one photo, using a public detection dataset.

### Recommender systems

**The job:** deciding what each user sees: products, content, offers. Huge commercial impact, with its own evaluation problems (feedback loops, popularity bias).
**Learn next:** matrix factorisation, implicit feedback, two-tower retrieval models, ranking metrics (NDCG, MAP), online A/B testing.
**Free resources:**
- Google, [*Recommendation Systems* course](https://developers.google.com/machine-learning/recommendation)
- [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) repository and notebooks
- The [MovieLens datasets](https://files.grouplens.org/datasets/movielens/)
**First project:** build an implicit-feedback recommender on Online Retail II and beat the item-item method from 06.10 on hit rate @10, evaluated on each customer's last purchase.

### Time series and forecasting

**The job:** demand, capacity, finance, energy. Very practical, with a strong statistical tradition and a lot of value in honest backtesting.
**Learn next:** hierarchical forecasting (product → category → total), probabilistic forecasts, global models across many series, intermittent demand.
**Free resources:**
- Rob Hyndman and George Athanasopoulos, [*Forecasting: Principles and Practice*](https://otexts.com/fpp3) (free online)
- [Nixtla](https://nixtlaverse.nixtla.io) open-source forecasting libraries and docs
- [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series)
**First project:** forecast weekly demand for each of ShopWave's top 50 products, reconcile them so they add up to the total, and compare against seasonal naive per product.

### Research and the maths path

**The job:** understanding and inventing methods, reading papers, often in a research lab or a PhD. Requires the maths you met by hand in this course, done properly.
**Learn next:** linear algebra (vectors, matrices, eigenvalues: PCA and attention), probability and statistics (likelihood, Bayes: logistic regression and Naive Bayes), calculus (derivatives and the chain rule: gradient descent and backpropagation), then optimisation.
**Free resources:**
- 3Blue1Brown, [*Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra) and [*Essence of Calculus*](https://www.3blue1brown.com/topics/calculus)
- Deisenroth, Faisal and Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io) (free PDF)
- [Stanford CS229](https://cs229.stanford.edu) lecture notes
- Kevin Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html) (free PDF)
**First project:** implement logistic regression, a small neural network and K-Means from scratch in NumPy, derive each gradient on paper first, and check every result against scikit-learn or Keras (you did the first steps in 02.4, 08.4 and 06.2).

## 4. How to keep up without drowning

- **Pick one newsletter or digest** and read it weekly, not daily. Ignore the rest.
- **Read one paper a month properly**, ideally one behind a tool you use. Reproduce one figure.
- **Re-implement, don't just read.** One small notebook beats ten bookmarked threads.
- **Join one community** (a Kaggle competition, a local meetup, an open-source project's issue tracker) and contribute something small.
- **Revisit your old projects.** A year from now, rebuild your capstone with what you have learned since. The difference is your progress, made visible.

## 5. A note on certificates

Certificates from cloud providers (AWS, Google Cloud, Azure ML) can help you get past a CV filter, especially for MLOps and platform roles, and their study material is a decent tour of the provider's tools. They are optional. A clear portfolio of three to five honest projects, each with a README that a reviewer understands in two minutes, does more for you in an interview than any certificate.

## 6. Closing

When you started, ShopWave had a spreadsheet and a CEO who kept reading about AI. Now it has sales reports, a spend model, a churn model with a cost-based threshold, customer segments, a Christmas forecast, a product image tagger, a support assistant that says "I don't know" when it should, and someone who can explain all of it, including where each one fails.

That last part matters most. The tools will change; most of the libraries in this course will have new versions next year. Framing a problem carefully, refusing to fool yourself with a leaky split, comparing against a simple baseline and saying plainly what a model cannot do: those habits do not go out of date.

Good luck. Go and build something someone will use.

**Back to the start:** [course README](../../../README.md) · [ROADMAP](../../../ROADMAP.md)
