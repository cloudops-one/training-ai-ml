# Glossary

One line per term. The lesson where it is first taught is in brackets. Terms are grouped, not alphabetical, so related ideas sit together.

## The basics

- **Artificial intelligence (AI)**: any program that does something we would call "smart", whether by rules or by learning. [00.2]
- **Machine learning (ML)**: programs that improve at a task by learning patterns from data instead of following hand-written rules. [00.1]
- **Deep learning (DL)**: machine learning using neural networks with many layers. [00.2]
- **Generative AI**: models that create new content (text, images, code), such as large language models. [00.2]
- **Model**: a function, learned from data, that turns inputs into a prediction. [02.1]
- **Training**: the process of adjusting a model's internal numbers so its predictions get better on the training data. [02.1]
- **Feature**: an input column the model uses to predict, e.g. `age`, `last_purchase_days`. Also called a variable or predictor. Written **X**. [02.1]
- **Target**: the column you want to predict, e.g. `spend_next_month`. Also called the label or outcome. Written **y**. [02.1]
- **Prediction**: the model's output for one row. Written **ŷ** ("y-hat"). [02.1]
- **Parameter**: a number inside the model that training changes, e.g. the slope of a line. [02.2]
- **Hyperparameter**: a setting you choose *before* training, e.g. the number of neighbours in KNN. [03.2]
- **Supervised learning**: learning from examples that include the correct answer (target). [00.3]
- **Unsupervised learning**: finding structure in data that has no target. [00.3]
- **Reinforcement learning**: learning by trial and error from rewards. [00.3]
- **Regression**: predicting a number. [02.1]
- **Classification**: predicting a category. [03.1]
- **Clustering**: grouping similar rows without labels. [06.1]

## Data

- **DataFrame**: a table in Pandas: rows and named columns. [01.3]
- **Categorical feature**: takes one of a few values (`country`, `plan_type`). [01.5]
- **Numerical feature**: a quantity (`price`, `age`). [01.5]
- **Missing value**: an empty cell, shown as `NaN`. [01.5]
- **Outlier**: a value far from the rest, which may be an error or a genuinely rare case. [01.8]
- **Imputation**: filling missing values with a sensible guess (mean, median, most common). [04.3]
- **Encoding**: turning categories into numbers so a model can use them. [04.4]
- **One-hot encoding**: one new 0/1 column per category. [04.4]
- **Scaling / standardisation**: rescaling numbers so features have comparable ranges. [04.5]
- **Feature engineering**: creating better input columns from the raw ones. [04.6]
- **EDA (exploratory data analysis)**: looking at the data with tables and charts before modelling. [01.9]

## Statistics

- **Mean**: the average. Sum ÷ count. [01.8]
- **Median**: the middle value when sorted. Robust to outliers. [01.8]
- **Standard deviation (σ)**: typical distance of values from the mean. [01.8]
- **Variance**: standard deviation squared. [01.8]
- **Percentile**: the value below which a given % of the data falls. The 25th, 50th, 75th are called quartiles. [01.8]
- **IQR (interquartile range)**: 75th percentile minus 25th; used to flag outliers. [01.8]
- **Correlation (r)**: how strongly two numbers move together, from −1 to +1. [01.8]
- **Distribution**: the shape of how values are spread, e.g. bell-shaped (normal) or skewed. [01.8]

## Training and evaluating

- **Training set**: the rows the model learns from. [04.1]
- **Test set**: rows hidden during training, used once to measure real performance. [04.1]
- **Validation set**: rows used during development to compare settings, keeping the test set untouched. [04.1]
- **Data leakage**: when information from the test set (or the future) sneaks into training, giving falsely good scores. [04.2]
- **Cross-validation**: splitting data into *k* folds and training *k* times so every row is tested once. [04.7]
- **Loss / cost function**: a number that measures how wrong the model is; training tries to make it small. [02.4]
- **Gradient descent**: repeatedly nudging parameters in the direction that reduces the loss. [02.4]
- **Learning rate**: the size of each nudge in gradient descent. [02.4]
- **Epoch**: one full pass over the training data. [08.6]
- **Overfitting**: the model memorises the training data and does badly on new data. [02.6]
- **Underfitting**: the model is too simple to capture the pattern. [04.9]
- **Bias–variance trade-off**: simple models err by being too rigid (bias); complex ones by being too sensitive to the training set (variance). [04.9]
- **Regularisation**: a penalty on large parameters that discourages overfitting. [04.10]
- **Pipeline**: preprocessing steps and a model chained into one object that is fitted and applied together. [04.8]
- **Baseline**: the simplest possible model (predict the mean, predict the majority class) that any real model must beat. [02.3]

## Regression metrics

- **MAE (mean absolute error)**: average size of the mistakes, in the target's units. [02.3]
- **MSE (mean squared error)**: average of squared mistakes; punishes big errors. [02.3]
- **RMSE**: square root of MSE, back in the target's units. [02.3]
- **R² (R-squared)**: share of the target's variation the model explains; 1 is perfect, 0 is no better than the mean. [02.3]

## Classification metrics

- **Confusion matrix**: a 2×2 table of true positives, false positives, false negatives, true negatives. [03.6]
- **Accuracy**: fraction of predictions that were right. Misleading when classes are imbalanced. [03.6]
- **Precision**: of the rows predicted positive, how many really were. [03.6]
- **Recall (sensitivity)**: of the rows that really are positive, how many we caught. [03.6]
- **F1 score**: harmonic mean of precision and recall. [03.6]
- **Threshold**: the probability above which we call a prediction "positive"; default 0.5, but it is a business choice. [03.7]
- **ROC curve / AUC**: recall vs false-positive rate at every threshold; AUC is the area beneath it, 1 is perfect, 0.5 is random. [03.7]
- **Class imbalance**: when one class is much rarer than the other (e.g. 2% fraud). [05.7]

## Models

- **Linear regression**: predicts with a straight line (or flat plane) through the data. [02.2]
- **Logistic regression**: linear model squashed through a sigmoid to give a probability. [03.4]
- **Sigmoid**: the S-shaped function 1 / (1 + e^(−z)) that maps any number to 0–1. [03.4]
- **KNN (k-nearest neighbours)**: predicts by looking at the *k* most similar training rows. [03.2]
- **Decision tree**: a flow-chart of yes/no questions learned from data. [02.7]
- **Gini impurity / entropy**: how mixed the classes are in a group; trees pick splits that reduce it. [03.3]
- **Ensemble**: several models combined. [05.1]
- **Bagging**: training many models on random samples and averaging them. [05.2]
- **Random forest**: bagging with decision trees plus random feature choice. [05.2]
- **Boosting**: training models one after another, each fixing the previous one's errors. [05.3]
- **Gradient boosting / XGBoost / LightGBM**: fast, strong boosting implementations that dominate tabular data. [05.4]
- **SVM (support vector machine)**: finds the boundary with the widest margin between classes. [05.5]
- **Naive Bayes**: classifies by multiplying probabilities of each feature given the class. [03.5]
- **K-Means**: clustering by repeatedly assigning points to the nearest centre and moving centres. [06.2]
- **Centroid**: the centre (mean) of a cluster. [06.2]
- **Silhouette score**: how well each point fits its cluster vs the next nearest; used to pick *k*. [06.3]
- **DBSCAN**: clustering by density; finds odd shapes and flags noise. [06.5]
- **PCA (principal component analysis)**: compresses many features into a few new ones that keep most of the variation. [06.6]
- **Isolation Forest**: anomaly detection by how easily a point can be isolated with random splits. [06.8]
- **Support / confidence / lift**: how often items appear together, how often B follows A, and how much more than chance. [06.9]
- **Collaborative filtering**: recommending based on what similar users liked. [06.10]
- **SHAP values**: how much each feature pushed one prediction up or down. [05.8]

## Time series

- **Trend**: long-run direction. **Seasonality**: repeating pattern. **Noise**: the rest. [07.1]
- **Stationarity**: statistical properties do not change over time; many models assume it. [07.4]
- **Lag feature**: yesterday's value used as a feature for today. [07.5]
- **ARIMA / SARIMA**: classical models that forecast from past values and past errors (S adds seasonality). [07.4]
- **Backtesting**: evaluating a forecaster by rolling through time, never peeking ahead. [07.2]

## Deep learning

- **Neuron**: weighted sum of inputs plus a bias, passed through an activation function. [08.2]
- **Weight / bias**: the parameters of a neuron. [08.2]
- **Activation function**: a non-linear squash (ReLU, sigmoid, tanh, softmax) applied after the weighted sum. [08.3]
- **Layer**: a group of neurons working in parallel. **Hidden layer**: any layer between input and output. [08.2]
- **Backpropagation**: computing how much each weight contributed to the error, so gradient descent can fix it. [08.4]
- **Batch**: the number of rows processed before one weight update. [08.6]
- **Dropout**: randomly switching off neurons during training to reduce overfitting. [08.6]
- **Early stopping**: halting training when validation loss stops improving. [08.6]
- **CNN (convolutional neural network)**: a network that slides small filters over an image to detect patterns. [08.7]
- **Convolution / filter / kernel**: a small grid of weights multiplied over patches of the image. [08.7]
- **Pooling**: shrinking a feature map by keeping the max or mean of each patch. [08.7]
- **Transfer learning**: reusing a network trained on a huge dataset and adapting it to yours. [08.9]
- **RNN / LSTM**: networks with memory for sequences; mostly replaced by Transformers. [08.10]

## Language and generative AI

- **Token**: a chunk of text (roughly a word or part of a word) that a model reads. [09.1]
- **Bag-of-words / TF-IDF**: representing text as word counts, optionally weighted by how rare the word is. [09.1]
- **Embedding**: a list of numbers representing the meaning of a word, sentence or document. [09.3]
- **Cosine similarity**: how close two embeddings point in the same direction, from −1 to 1. [09.3]
- **Attention**: a mechanism that lets each token look at the other tokens and weigh them. [09.4]
- **Transformer**: the architecture built on attention that powers BERT, GPT and modern LLMs. [09.4]
- **Pretrained model**: a model already trained on huge data; you reuse or fine-tune it. [09.5]
- **Fine-tuning**: continuing training a pretrained model on your own data. [09.6]
- **LLM (large language model)**: a very large Transformer trained to predict the next token. [09.7]
- **Context window**: the maximum number of tokens an LLM can read at once. [09.7]
- **Temperature**: a knob for randomness in generation; 0 is deterministic. [09.7]
- **Prompt**: the text you give an LLM. **System prompt**: standing instructions. [09.8]
- **Few-shot**: including examples in the prompt. **Zero-shot**: none. [09.8]
- **Hallucination**: a fluent but false statement from an LLM. [09.11]
- **Prompt injection**: malicious text that tries to override your instructions. [09.11]
- **RAG (retrieval-augmented generation)**: fetching relevant documents and placing them in the prompt so the LLM answers from your data. [09.10]
- **Vector database**: a store that finds embeddings similar to a query quickly. [09.10]
- **Chunking**: splitting documents into pieces small enough to embed and retrieve. [09.10]
- **Agent**: an LLM that decides which tools to call and loops until the task is done. [09.12]
- **Tool calling / function calling**: letting an LLM request that your code runs a function. [09.12]

## Production

- **Deployment**: making a model available to users or other systems. [10.1]
- **API**: a web endpoint that takes inputs and returns predictions. [10.2]
- **Docker container**: a packaged environment that runs the same anywhere. [10.4]
- **Experiment tracking**: logging parameters, metrics and artifacts of every training run. [10.5]
- **Data drift**: the input data changes over time. **Concept drift**: the relationship between inputs and target changes. [10.6]
- **MLOps**: the practices for building, deploying and maintaining ML systems reliably. [10.5]
- **Responsible AI**: fairness, privacy, transparency and accountability in ML systems. [10.7]
