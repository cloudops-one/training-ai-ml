# 11.1 · Choosing and scoping a capstone problem

⏱ about 1 hour · Prerequisites: Modules 00–10 · Worksheet: [scoping_template.ipynb](scoping_template.ipynb)

---

## 1. The problem

The most common way a capstone fails is not a bad model. It is a problem that was never scoped: "predict the stock market", "build a chatbot for my company" (no data), "detect fraud" (no labels), "something with images" (no question). Weeks pass, the notebook grows, nothing finishes.

This lesson takes an hour and saves the other fourteen.

## 2. The idea

A good capstone is the smallest problem that still shows **every step of the workflow** from 00.4, on data you can actually get, with a result you can compare against something simpler.

Think of it as a short story, not a novel: one character (a user), one decision, one clear ending (did the model help, yes or no, by how much).

## 3. The four tests

Before you commit to a problem, it must pass all four.

| Test | Question to ask | Fails when… |
|---|---|---|
| **Data** | Can I download it today, legally, with a licence that allows a public repo? | It needs a login you cannot share, scraping that breaks terms of service, or private company data. |
| **Decision** | Who would do something differently because of my model? | Nobody: the output is "interesting" but changes no action. |
| **Baseline** | What is the simplest thing that could work, and can I measure it? | You cannot say what "better" means, or there is no simpler comparison. |
| **Size** | Can I get a first end-to-end result in about 5 hours and finish in about 15? | The data is 200 GB, needs a GPU cluster, or you would have to label it yourself. |

## 4. The scoping worksheet

Fill this in before writing any modelling code. It is also in [scoping_template.ipynb](scoping_template.ipynb) so you can keep it next to your work.

| Field | Your answer, in one or two sentences |
|---|---|
| **Problem statement** | What is going wrong, or what could be better, for whom? |
| **User** | The person or team who will use the output. |
| **Decision** | What they will do differently with your output. |
| **Target** | Exactly what you predict, cluster, forecast or generate, and over what time window. |
| **Unit** | One row of your modelling table is one… (customer, order, day, image, document). |
| **Data** | Source, licence, size, date range, how you will download it. |
| **Baseline** | The simplest rule or model you must beat. |
| **Metric** | The number you will report, and why it fits the decision (03.7, 05.10). |
| **Risks** | Leakage, imbalance, drift, missing data, fairness, privacy. |
| **Definition of done** | What "finished" looks like, so you know when to stop. |

## 5. A worked example: ShopWave's premium catalogue

This is the scoping sheet behind the [worked example capstone](../02-capstone/example/).

| Field | Answer |
|---|---|
| Problem statement | The premium catalogue costs about £3 per copy and goes to 500 customers a quarter. Many copies go to customers who then spend little. |
| User | ShopWave's marketing team. |
| Decision | Which 500 customers receive the catalogue each quarter. |
| Target | 1 if a customer spends £500 or more in the three months after a cut-off date, else 0. |
| Unit | One registered customer, as of a cut-off date. |
| Data | Online Retail II, UCI, CC BY 4.0; 2010–2011 sheet; downloaded by `data/download.py`. |
| Baseline | Today's rule: the 500 customers with the highest spend in the last 90 days. |
| Metric | Precision among the top 500 (the budget), with ROC-AUC for comparing models. |
| Risks | Leakage across the cut-off; seasonality (the autumn quarter is bigger); customers with one order. |
| Definition of done | A time-based backtest showing whether a model beats the rule for the same 500 catalogues, a saved model, a model card, a monitoring plan. |

Notice how small the problem is. One target, one dataset, one decision. That is a feature, not a weakness.

## 6. Project ideas

Each idea below passes the four tests. Pick one, change one, or bring your own and run it through the worksheet. All datasets are free to download; check each licence before you publish.

**Level 1: closest to what you have done**

1. **ShopWave, continued.** Pick a question the course did not answer on [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): predict which products a customer will reorder; forecast weekly demand for the top 20 products; detect returns. Both sheets give you two years.
2. **Bike-sharing demand.** Forecast hourly rentals from weather and calendar features with the [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) data. A clean forecasting problem with a clear operational decision (how many bikes to move).
3. **Telco churn.** The public [IBM Telco customer churn](https://github.com/IBM/telco-customer-churn-on-icp4d) data. Build a churn model with a cost-based threshold, as in 03.7 and 05.10, and explain it with SHAP.

**Level 2: a new domain, same toolkit**

4. **E-commerce delivery delays.** The [Olist Brazilian e-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) data (free Kaggle account, CC BY-NC-SA). Predict late deliveries or low review scores from order, seller and geography features.
5. **Taxi trip duration or fare.** A month of [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Regression with real outliers and a lot of feature engineering. Sample it: a month is millions of rows.
6. **Product image classifier.** Fine-tune a pretrained CNN (08.9) on a public product-image set such as [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) to predict category or colour. Report per-class errors and a confidence threshold for automatic tagging.

**Level 3: language and generative AI**

7. **Support-ticket router.** Classify public customer-support messages (for example the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)) into intents. Compare TF-IDF + logistic regression, a fine-tuned small transformer and an LLM prompt, on accuracy, cost and latency (Module 09).
8. **RAG assistant over public documentation.** Build a question-answering assistant over a public, permissively licensed document set (a city's open-data portal guides, an open-source project's docs, a government service's published FAQs). Build a 30-question evaluation set first, then the system (09.10, 09.11).

## 7. Red flags

Stop and rescope if you hear yourself say:

- "I'll find the data later."
- "The target is whatever the model finds interesting." (Unsupervised work still needs a decision it supports.)
- "Accuracy will be the metric" on a problem where one class is 2% of the data (05.7).
- "I'll use every column." (Some of them are from the future; 04.2.)
- "First I need to label 5,000 examples."
- "It only works on a GPU cluster."
- "It's a chatbot that can answer anything." (Pick ten kinds of question and evaluate those.)
- "It predicts stock prices." (Your baseline, "tomorrow equals today", is very hard to beat, and nobody will act on your model.)

## 8. Check yourself

1. Fill in the worksheet for your chosen problem. Every row, one or two sentences.
2. Show it to someone who has not taken this course. Can they say, in their own words, who will use the output and what they will do with it?
3. Download the data and load it. If that takes more than an hour, the Size test has failed: pick something else now, not in week three.

## 9. Recap

- A capstone must pass four tests: **data, decision, baseline, size**.
- Write the scoping sheet before any modelling code. It is a contract with yourself.
- Small, finished and honest beats ambitious and half-done.
- The worked example shows how small a good problem can be.

**Next:** [11.2 The capstone](../02-capstone/) gives you the brief, the rubric, a starter template and a worked example to compare against.
