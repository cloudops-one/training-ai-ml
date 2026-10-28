# Roadmap

Every lesson in order. Tick them off as you go. Hours are estimates for reading, running the notebook and doing the exercises; your speed is your own.

Legend: **core** = do not skip · *optional* = useful, skip if short on time.

---

## Module 00 · The Big Picture (4 h)

*ShopWave asks: "Can machine learning even help us?"*

- [ ] 00.1 Why machine learning? Rules vs learning from data (1 h) **core**
- [ ] 00.2 The AI family tree: AI, ML, deep learning, generative AI (0.5 h) **core**
- [ ] 00.3 The three kinds of learning: supervised, unsupervised, reinforcement (0.5 h) **core**
- [ ] 00.4 The 7-step ML workflow (0.5 h) **core**
- [ ] 00.5 The whole course in one notebook: a 30-minute tour from raw data to a deployed model (1.5 h) **core**

## Module 01 · Working with Data (20 h)

*ShopWave asks: "Make sense of our sales data."*

- [ ] 01.1 NumPy: arrays, shapes and why vectors beat loops (2 h) **core**
- [ ] 01.2 NumPy: indexing, boolean masks, broadcasting, random numbers (2 h) **core**
- [ ] 01.3 Pandas: Series, DataFrames, reading files, inspecting data (2 h) **core**
- [ ] 01.4 Pandas: selecting, filtering, sorting, new columns (2 h) **core**
- [ ] 01.5 Pandas: missing values, duplicates, data types, dates (2 h) **core**
- [ ] 01.6 Pandas: groupby, merge, pivot (2 h) **core**
- [ ] 01.7 Plotting with Matplotlib and Seaborn: the six charts you actually need (2 h) **core**
- [ ] 01.8 Statistics you need: mean, median, spread, percentiles, correlation, by hand (2 h) **core**
- [ ] 01.9 Exploratory data analysis: a repeatable checklist (2 h) **core**
- [ ] 01.P Mini-project: the ShopWave sales report (2 h) **core**

## Module 02 · Regression (14 h)

*ShopWave asks: "How much will a customer spend next month?"*

- [ ] 02.1 What is a model? Features, target, prediction, error (1 h) **core**
- [ ] 02.2 Linear regression by hand: the best straight line (2 h) **core**
- [ ] 02.3 Measuring error: MAE, MSE, RMSE, R² (1.5 h) **core**
- [ ] 02.4 Gradient descent: how a computer finds the line by walking downhill (2 h) **core**
- [ ] 02.5 Many features: multiple linear regression (1.5 h) **core**
- [ ] 02.6 Curves: polynomial regression and the first look at overfitting (1.5 h) **core**
- [ ] 02.7 Decision tree regressor: predicting with questions (1.5 h) **core**
- [ ] 02.P Mini-project: predict customer spend (3 h) **core**

## Module 03 · Classification (16 h)

*ShopWave asks: "Which customers are about to leave?"*

- [ ] 03.1 Classification: predicting a label instead of a number (1 h) **core**
- [ ] 03.2 K-nearest neighbours by hand (1.5 h) **core**
- [ ] 03.3 Decision tree classifier: Gini impurity by hand (2 h) **core**
- [ ] 03.4 Logistic regression: the sigmoid and probabilities (2 h) **core**
- [ ] 03.5 Naive Bayes: counting your way to a prediction (1.5 h) *optional*
- [ ] 03.6 The confusion matrix: accuracy, precision, recall, F1 (2 h) **core**
- [ ] 03.7 ROC curves, AUC and choosing a threshold by business cost (1.5 h) **core**
- [ ] 03.8 Multi-class classification (1 h) **core**
- [ ] 03.P Mini-project: Titanic survival, then ShopWave churn (3.5 h) **core**

## Module 04 · Making Models Trustworthy (14 h)

*ShopWave asks: "It worked in the notebook. Why is it wrong in production?"*

- [ ] 04.1 Train, validation, test: why you must hide data from yourself (1 h) **core**
- [ ] 04.2 Data leakage: the mistake that makes every model look great (1 h) **core**
- [ ] 04.3 Missing values and outliers done properly (1.5 h) **core**
- [ ] 04.4 Encoding categories: one-hot, ordinal, target encoding (1.5 h) **core**
- [ ] 04.5 Scaling features: who needs it and who does not (1 h) **core**
- [ ] 04.6 Feature engineering: dates, ratios, bins, RFM (1.5 h) **core**
- [ ] 04.7 Cross-validation: a fairer score (1.5 h) **core**
- [ ] 04.8 Pipelines and ColumnTransformer: leak-proof preprocessing (1.5 h) **core**
- [ ] 04.9 Overfitting, underfitting and the bias–variance trade-off (1.5 h) **core**
- [ ] 04.10 Regularisation: Ridge and Lasso (1 h) **core**
- [ ] 04.P Mini-project: rebuild churn as a leak-free pipeline (1 h) **core**

## Module 05 · Powerful Models & Explainability (16 h)

*ShopWave asks: "Make it as accurate as possible, then explain it to the CEO."*

- [ ] 05.1 Ensembles: why many weak models beat one strong one (1 h) **core**
- [ ] 05.2 Bagging and Random Forest (1.5 h) **core**
- [ ] 05.3 Boosting: AdaBoost and gradient boosting by hand (2 h) **core**
- [ ] 05.4 XGBoost, LightGBM and CatBoost in practice (1.5 h) **core**
- [ ] 05.5 Support vector machines (1.5 h) *optional*
- [ ] 05.6 Hyperparameter tuning: grid, random, Bayesian (Optuna) (1.5 h) **core**
- [ ] 05.7 Imbalanced data: class weights, resampling, SMOTE, the right metrics (1.5 h) **core**
- [ ] 05.8 Explaining models: feature importance, permutation importance, SHAP (2 h) **core**
- [ ] 05.9 AutoML in industry: PyCaret and what it automates (1 h) *optional*
- [ ] 05.10 Choosing and saving the final model (0.5 h) **core**
- [ ] 05.P Mini-project: a tuned, explained churn model (2 h) **core**

## Module 06 · Patterns Without Labels (14 h)

*ShopWave asks: "Group our customers. Find fraud. What sells together?"*

- [ ] 06.1 Unsupervised learning: what you can do without a target (0.5 h) **core**
- [ ] 06.2 K-Means by hand (2 h) **core**
- [ ] 06.3 How many clusters? Elbow, silhouette, and profiling clusters (1.5 h) **core**
- [ ] 06.4 Hierarchical clustering and dendrograms (1 h) **core**
- [ ] 06.5 DBSCAN and Gaussian mixtures (1.5 h) **core**
- [ ] 06.6 PCA: fewer dimensions, same story (2 h) **core**
- [ ] 06.7 t-SNE and UMAP for pictures of high-dimensional data (0.5 h) *optional*
- [ ] 06.8 Anomaly detection: z-score, IQR, Isolation Forest (1.5 h) **core**
- [ ] 06.9 Market basket analysis: support, confidence, lift (1.5 h) **core**
- [ ] 06.10 Recommender systems: content-based and collaborative filtering (1 h) **core**
- [ ] 06.P Mini-project: RFM customer segments and fraud flags (1 h) **core**

## Module 07 · Forecasting (10 h)

*ShopWave asks: "How much stock do we need for December?"*

- [ ] 07.1 Time series: trend, seasonality, noise, decomposition (1.5 h) **core**
- [ ] 07.2 Baselines and honest evaluation: time-ordered splits, MAPE (1.5 h) **core**
- [ ] 07.3 Smoothing and Holt-Winters (1.5 h) **core**
- [ ] 07.4 ARIMA and SARIMA (2 h) **core**
- [ ] 07.5 Forecasting with machine learning: lag features and gradient boosting (1.5 h) **core**
- [ ] 07.P Mini-project: monthly sales forecast with backtesting (2 h) **core**

## Module 08 · Deep Learning (24 h)

*ShopWave asks: "Auto-tag our product photos."*

- [ ] 08.1 When deep learning wins, and when it does not (0.5 h) **core**
- [ ] 08.2 One neuron, by hand and in NumPy (2 h) **core**
- [ ] 08.3 Activation functions and loss functions (1.5 h) **core**
- [ ] 08.4 Backpropagation on a two-layer network, by hand (3 h) **core**
- [ ] 08.5 Your first network in Keras (2 h) **core**
- [ ] 08.6 Training well: epochs, batches, learning rate, dropout, early stopping (2 h) **core**
- [ ] 08.7 Images as numbers; convolution by hand on a 5×5 image (2 h) **core**
- [ ] 08.8 CNNs in Keras on Fashion-MNIST (2 h) **core**
- [ ] 08.9 Data augmentation and transfer learning (2.5 h) **core**
- [ ] 08.10 Sequences: RNNs, LSTMs and why they led to Transformers (1.5 h) **core**
- [ ] 08.11 The same model in PyTorch (1.5 h) *optional*
- [ ] 08.P Mini-project: ShopWave product image classifier (3.5 h) **core**

## Module 09 · Language & Generative AI (24 h)

*ShopWave asks: "Read 50,000 reviews. Answer support chats."*

- [ ] 09.1 Text to numbers: cleaning, tokens, bag-of-words, TF-IDF by hand (2 h) **core**
- [ ] 09.2 Sentiment classifier on product reviews (1.5 h) **core**
- [ ] 09.3 Embeddings: meaning as a vector, cosine similarity by hand (2 h) **core**
- [ ] 09.4 Attention and the Transformer, explained with a 4-word sentence (2.5 h) **core**
- [ ] 09.5 Hugging Face: pretrained models in three lines (1.5 h) **core**
- [ ] 09.6 Fine-tuning a small model on ShopWave reviews (2 h) **core**
- [ ] 09.7 How large language models work: tokens, context, temperature, training stages (1.5 h) **core**
- [ ] 09.8 Prompt engineering and structured output (2 h) **core**
- [ ] 09.9 Calling an LLM from Python: APIs, cost, safety, local fallback (1.5 h) **core**
- [ ] 09.10 Retrieval-augmented generation: chunk, embed, retrieve, generate (3 h) **core**
- [ ] 09.11 Evaluating LLM outputs; hallucination, bias, prompt injection (1 h) **core**
- [ ] 09.12 Agents and tool calling (1.5 h) **core**
- [ ] 09.P Mini-project: review analyser and ShopWave support chatbot (2 h) **core**

## Module 10 · Shipping & Running ML (10 h)

*ShopWave asks: "Put it in production and keep it working."*

- [ ] 10.1 Saving and loading models; reproducibility (0.5 h) **core**
- [ ] 10.2 A prediction API with FastAPI (1.5 h) **core**
- [ ] 10.3 A web app with Streamlit (1 h) **core**
- [ ] 10.4 Docker in one hour (1 h) *optional*
- [ ] 10.5 Experiment tracking with MLflow (1.5 h) **core**
- [ ] 10.6 Monitoring, data drift, concept drift, retraining (1.5 h) **core**
- [ ] 10.7 Responsible AI: fairness, privacy, explainability, governance (1 h) **core**
- [ ] 10.P Mini-project: deploy the churn model and the chatbot (2 h) **core**

## Module 11 · Capstone & Next Steps (20 h)

- [ ] 11.1 Choosing and scoping a capstone problem (1 h) **core**
- [ ] 11.2 The capstone: end-to-end, with rubric (15 h) **core**
- [ ] 11.3 Writing it up: README, portfolio, presentation (2 h) **core**
- [ ] 11.4 Interview preparation: 60 questions with model answers (1 h) **core**
- [ ] 11.5 What to learn next: specialisation map (1 h) **core**
