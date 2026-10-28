# 11.4 · Interview preparation: 60 questions with model answers

⏱ about 1 hour (then revisit before every interview) · Prerequisites: Modules 00–10

---

## How to use this page

Cover each answer and say yours out loud first. Then compare. The model answers are deliberately short: in an interview, a clear two-sentence answer followed by an example from **your own project** beats a memorised paragraph. Every answer links to the lesson that teaches it; if an answer surprises you, go back there.

The interviewer is checking three things: do you understand the idea, can you explain it simply, and have you used it.

---

## Fundamentals

**1. What is machine learning, and when should you not use it?**
Learning a function from examples (data plus answers) instead of writing rules by hand. Don't use it when the rule is known and fixed, when a guarantee is required, or when there is no data. → [00.1](../../00-big-picture/01-why-machine-learning/)

**2. Supervised vs unsupervised vs reinforcement learning?**
Supervised learns from labelled examples (churn yes/no); unsupervised finds structure without labels (customer segments); reinforcement learns actions from rewards (which discount to offer). → [00.3](../../00-big-picture/03-three-kinds-of-learning/)

**3. Regression vs classification?**
Regression predicts a number (next month's spend), classification a category (will churn or not). Logistic regression, despite its name, is a classifier. → [02.1](../../02-regression/01-what-is-a-model/), [03.1](../../03-classification/01-what-is-classification/)

**4. Parameter vs hyperparameter?**
Parameters are learned during training (a line's slope, a network's weights). Hyperparameters are chosen before training (k in KNN, tree depth, learning rate), usually by cross-validation. → [02.1](../../02-regression/01-what-is-a-model/), [05.6](../../05-powerful-models/06-hyperparameter-tuning/)

**5. Describe the ML workflow.**
Frame the problem, get the data, explore it, prepare features, train against a baseline, evaluate honestly, deploy and monitor. Most failures happen in framing and evaluation, not in modelling. → [00.4](../../00-big-picture/04-ml-workflow/)

**6. What is gradient descent?**
Repeatedly nudge parameters in the direction that lowers the loss, by a step size called the learning rate. Too large and it diverges; too small and it crawls. → [02.4](../../02-regression/04-gradient-descent/)

**7. What is overfitting and how do you detect it?**
The model memorises training noise: great training score, worse validation score. Detect it with a held-out set or learning curves; fix it with more data, a simpler model, regularisation or early stopping. → [02.6](../../02-regression/06-polynomial-and-overfitting/), [04.9](../../04-trustworthy-models/09-bias-variance/)

**8. Explain the bias–variance trade-off.**
Simple models err by being too rigid (high bias); flexible models err by reacting to the particular training sample (high variance). Expected error is bias² + variance + noise; you tune complexity to minimise the sum. → [04.9](../../04-trustworthy-models/09-bias-variance/)

## Data and features

**9. How do you handle missing values?**
First ask why they are missing. Then impute (median, most frequent, or a model), add a "was missing" indicator when missingness itself is informative, and fit the imputer on training data only. → [04.3](../../04-trustworthy-models/03-missing-and-outliers/)

**10. How do you treat outliers?**
Decide whether each is an error or a real rare case. Errors get fixed or dropped; real cases are kept, capped or handled with a log transform or a robust model. Never drop them just because they are inconvenient. → [04.3](../../04-trustworthy-models/03-missing-and-outliers/), [06.8](../../06-unsupervised/08-anomaly-detection/)

**11. One-hot vs label vs target encoding?**
One-hot for nominal categories with few values; ordinal encoding only when the order is real; target encoding for many categories, computed out-of-fold to avoid leakage. Label-encoding a nominal feature invents an order a linear model will use. → [04.4](../../04-trustworthy-models/04-encoding-categories/)

**12. Which models need feature scaling?**
Anything using distances or gradient-based optimisation with penalties: KNN, SVM, regularised linear models, neural networks, PCA. Tree-based models do not. → [04.5](../../04-trustworthy-models/05-scaling/)

**13. What is data leakage? Give an example.**
Information unavailable at prediction time sneaks into training. Example: a "days since last order" computed *after* the churn cut-off, which makes the model look 97% accurate in testing and fail live. → [04.2](../../04-trustworthy-models/02-data-leakage/)

**14. Give an example of good feature engineering.**
Recency, frequency and monetary value (RFM) from raw transactions; ratios like average order value; date parts; log transforms for skewed money columns. Domain knowledge usually beats a fancier model. → [04.6](../../04-trustworthy-models/06-feature-engineering/)

**15. Why use a Pipeline?**
It keeps preprocessing and model together, so cross-validation fits the preprocessing inside each fold (no leakage) and the same steps run on new data in production. → [04.8](../../04-trustworthy-models/08-pipelines/)

**16. What is the curse of dimensionality?**
With many features, points become nearly equidistant and data becomes sparse, so distance-based methods and density estimates weaken. Feature selection, PCA or regularisation help. → [06.6](../../06-unsupervised/06-pca/)

## Evaluation

**17. Why split into train, validation and test sets?**
Train to fit, validation to choose between models and settings, test once at the end for an honest estimate. Using the test set to choose makes its score optimistic. → [04.1](../../04-trustworthy-models/01-train-validation-test/)

**18. When should you split by time instead of randomly?**
Whenever the model will predict the future from the past: churn, demand, fraud. A random split lets future information leak into training. → [04.1](../../04-trustworthy-models/01-train-validation-test/), [07.2](../../07-forecasting/02-baselines-and-evaluation/)

**19. What is cross-validation and why use it?**
Split the training data into k folds, train k times, each time testing on a different fold. The mean and spread of the k scores are more reliable than one split's score. → [04.7](../../04-trustworthy-models/07-cross-validation/)

**20. Precision vs recall?**
Precision: of those flagged, how many were right. Recall: of the real positives, how many were caught. Choose by cost: missing fraud is expensive (favour recall); a wrongly blocked customer is expensive (favour precision). → [03.6](../../03-classification/06-confusion-matrix-metrics/)

**21. Why can accuracy be misleading?**
With 2% fraud, predicting "never fraud" scores 98% and catches nothing. Use precision, recall, F1 or PR-AUC, and always compare with the majority-class baseline. → [03.6](../../03-classification/06-confusion-matrix-metrics/), [05.7](../../05-powerful-models/07-imbalanced-data/)

**22. What does ROC-AUC mean?**
The probability that a randomly chosen positive is scored higher than a randomly chosen negative. 0.5 is random, 1.0 perfect. It measures ranking, not calibrated probabilities. → [03.7](../../03-classification/07-roc-auc-threshold/)

**23. How do you choose a classification threshold?**
Not 0.5 by default. Write down the cost of a false positive and of a false negative, compute expected cost at each threshold on validation data, and pick the minimum. → [03.7](../../03-classification/07-roc-auc-threshold/)

**24. MAE vs RMSE vs R² for regression?**
MAE is the typical error in the target's units and is robust to outliers; RMSE punishes large errors more; R² is the share of variance explained relative to predicting the mean. Report one in business units. → [02.3](../../02-regression/03-measuring-error/)

## Classical models

**25. How does linear regression find its line?**
It chooses the slope and intercept that minimise the sum of squared residuals; for one feature, slope = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)². → [02.2](../../02-regression/02-linear-regression-by-hand/)

**26. How does logistic regression produce a probability?**
A linear score passed through the sigmoid 1/(1+e^−z), trained by minimising log loss. Coefficients are changes in log-odds. → [03.4](../../03-classification/04-logistic-regression/)

**27. Ridge vs Lasso?**
Both add a penalty on coefficient size. Ridge (L2) shrinks all coefficients smoothly; Lasso (L1) can set some exactly to zero, acting as feature selection. → [04.10](../../04-trustworthy-models/10-regularisation/)

**28. How does a decision tree choose a split?**
It tries every feature and threshold and picks the one that most reduces impurity: Gini or entropy for classification, variance for regression. → [03.3](../../03-classification/03-decision-tree-classifier/), [02.7](../../02-regression/07-decision-tree-regressor/)

**29. Bagging vs boosting?**
Bagging trains many models independently on bootstrap samples and averages them, reducing variance (random forest). Boosting trains models in sequence, each fixing the previous ones' errors, reducing bias (gradient boosting). → [05.1](../../05-powerful-models/01-ensembles/), [05.3](../../05-powerful-models/03-boosting-by-hand/)

**30. Why does a random forest work better than a single tree?**
Averaging many decorrelated trees (random rows and random features per split) cancels much of each tree's variance while keeping their low bias. → [05.2](../../05-powerful-models/02-random-forest/)

**31. Why are XGBoost and LightGBM so popular on tabular data?**
Fast, strong gradient boosting with built-in regularisation, missing-value handling and early stopping. On the course's churn data they still only tied logistic regression: always keep the simple baseline. → [05.4](../../05-powerful-models/04-xgboost-lightgbm/)

**32. How does K-Means work, and how do you choose k?**
Assign points to the nearest centre, move each centre to its points' mean, repeat until stable. Choose k with the elbow and silhouette, then check the segments make business sense. → [06.2](../../06-unsupervised/02-kmeans-by-hand/), [06.3](../../06-unsupervised/03-choosing-k/)

## Deep learning

**33. What does a single neuron compute?**
A weighted sum of its inputs plus a bias, passed through an activation function. With a sigmoid activation it is exactly logistic regression. → [08.2](../../08-deep-learning/02-one-neuron/)

**34. Why do networks need non-linear activations?**
Stacked linear layers collapse into one linear layer. Non-linearities such as ReLU let the network represent curved decision boundaries. → [08.3](../../08-deep-learning/03-activations-and-losses/)

**35. Explain backpropagation simply.**
Run the inputs forward to get a loss, then use the chain rule backwards to compute how much each weight contributed to it, and update every weight with gradient descent. → [08.4](../../08-deep-learning/04-backpropagation-by-hand/)

**36. How do you stop a network overfitting?**
More data or augmentation, dropout, weight decay (L2), early stopping on validation loss, and a smaller network. Watch training and validation curves. → [08.6](../../08-deep-learning/06-training-well/)

**37. Why do CNNs suit images?**
Small filters slide over the image, sharing weights across positions, so they detect local patterns (edges, textures) anywhere with far fewer parameters than a dense layer. → [08.7](../../08-deep-learning/07-images-and-convolution/), [08.8](../../08-deep-learning/08-cnns-fashion-mnist/)

**38. What is transfer learning and when do you use it?**
Reusing a network pretrained on a huge dataset and retraining only its last layers on yours. Use it whenever your labelled data is small: in the course, 1,000 photos went from 55% from scratch to 85% with transfer. → [08.9](../../08-deep-learning/09-augmentation-transfer-learning/)

**39. When would you not use deep learning?**
On small or medium tabular data, where gradient boosting or logistic regression usually match it with less data, compute and explanation effort. → [08.1](../../08-deep-learning/01-when-deep-learning-wins/), [08.5](../../08-deep-learning/05-first-keras-network/)

## NLP and LLMs

**40. What is TF-IDF?**
Word counts weighted down for words common across all documents, so distinctive words count more. Fast, explainable and a strong baseline for text classification. → [09.1](../../09-nlp-genai/01-text-to-numbers/)

**41. What is an embedding?**
A vector representing meaning, where similar texts are close (high cosine similarity). It powers semantic search, clustering and RAG. → [09.3](../../09-nlp-genai/03-embeddings/)

**42. Explain self-attention in one minute.**
Each token builds a query, key and value. Its new representation is a weighted average of all tokens' values, weighted by how well its query matches their keys (softmax of QKᵀ/√d). Every token can look at every other token in parallel. → [09.4](../../09-nlp-genai/04-attention-transformers/)

**43. Fine-tuning vs prompting vs RAG?**
Prompting when the model already knows how; RAG when it needs your facts (policies, products) and they change; fine-tuning when you need a consistent style or task behaviour and have labelled examples. → [09.6](../../09-nlp-genai/06-fine-tuning/), [09.10](../../09-nlp-genai/10-rag/)

**44. What do temperature and the context window control?**
Temperature controls randomness when choosing the next token (0 is near-deterministic). The context window is the maximum number of tokens the model can read at once, including your prompt and documents. → [09.7](../../09-nlp-genai/07-how-llms-work/)

**45. Walk through a RAG pipeline.**
Chunk documents, embed the chunks, store them in a vector index; at question time embed the question, retrieve the top-k chunks, put them in the prompt with an instruction to answer only from them and cite, and say "I don't know" when retrieval finds nothing relevant. → [09.10](../../09-nlp-genai/10-rag/)

**46. How do you evaluate an LLM application?**
Build a question set with expected facts before building the system; measure retrieval hit rate and answer correctness automatically where possible; use an LLM judge only with a rubric and spot checks against human grading. → [09.11](../../09-nlp-genai/11-evaluating-llms/)

**47. What is prompt injection and how do you defend against it?**
Text in user input or retrieved documents that tries to override your instructions. Defend in layers: separate instructions from data, limit what tools can do, filter outputs in code, and never let the model's output trigger irreversible actions unchecked. → [09.11](../../09-nlp-genai/11-evaluating-llms/), [09.12](../../09-nlp-genai/12-agents-tool-calling/)

## MLOps

**48. How do you deploy a model?**
Save the whole pipeline, load it in a service (a FastAPI endpoint for real-time, a scheduled job for batch) that validates inputs, and version both the model and the code. → [10.1](../../10-deployment-mlops/01-saving-and-reproducibility/), [10.2](../../10-deployment-mlops/02-fastapi/)

**49. Batch vs real-time prediction?**
Batch scores many rows on a schedule (the quarterly catalogue list): simpler and cheaper. Real-time scores one request at a time (fraud at checkout): needed when the decision cannot wait. → [10.2](../../10-deployment-mlops/02-fastapi/)

**50. Data drift vs concept drift?**
Data drift: the inputs change (customers younger, baskets bigger). Concept drift: the relationship changes (the same customer now behaves differently). Monitor both inputs and, when labels arrive, performance. → [10.6](../../10-deployment-mlops/06-monitoring-drift/)

**51. Why track experiments?**
So every result can be traced to its code, data, parameters and metrics, compared fairly and reproduced. MLflow is a common tool. → [10.5](../../10-deployment-mlops/05-mlflow/)

**52. What goes in a model card?**
Intended use and misuse, data, evaluation (with baseline), limitations, fairness checks and a monitoring plan. → [05.10](../../05-powerful-models/10-choosing-and-saving/), [worked example](../02-capstone/example/model_card.md)

## Case questions

**53. Design a churn prediction system for an online store.**
Define churn (no order in 90 days after a cut-off); build a customer table with features only from before the cut-off; split by time; baseline "days since last order"; logistic regression and gradient boosting compared with CV; choose a threshold by voucher cost vs customer value; score weekly in batch; monitor precision in the contacted group and feature drift. → [Module 03 project](../../03-classification/project/), [04.P](../../04-trustworthy-models/project/), [05.P](../../05-powerful-models/project/)

**54. Your model has 99% accuracy. What do you check first?**
The class balance (is the majority baseline also 99%?), then leakage (a feature from the future or the label in disguise), then whether the split matches how it will be used. → [04.2](../../04-trustworthy-models/02-data-leakage/), [05.7](../../05-powerful-models/07-imbalanced-data/)

**55. A model's performance dropped in production. How do you investigate?**
Check the data pipeline first (broken feature, changed units, new categories), then input drift against training, then whether labels or behaviour changed (concept drift), then retrain on recent data and compare on the same holdout. → [10.6](../../10-deployment-mlops/06-monitoring-drift/)

**56. How would you forecast demand for the Christmas season?**
Weekly series with yearly seasonality; seasonal-naive baseline; rolling-origin backtest; compare Holt-Winters, SARIMA and gradient boosting with lag features; report error in units of stock, with intervals; plan stock from the upper interval for best-sellers. → [07.P](../../07-forecasting/project/)

## Behavioural, for ML roles

**57. Tell me about a project that didn't work.**
Use a real one. Say what you expected, what the honest evaluation showed, what you learned and what you changed. Example from the course: boosting did not beat logistic regression on churn, so the simpler model was kept. → [05.P](../../05-powerful-models/project/)

**58. How do you explain a model to a non-technical stakeholder?**
Start with the decision and the result in their units ("32 more high-value customers reached for the same budget"), then the two or three things that drive it, then where not to trust it. No jargon. → [05.8](../../05-powerful-models/08-explaining-models/), [11.3](../03-writing-it-up/)

**59. How do you decide when a model is good enough?**
When it beats the current approach by a margin that matters to the business, the evaluation is honest, the failure modes are understood and acceptable, and further gains would cost more than they return. → [05.10](../../05-powerful-models/10-choosing-and-saving/)

**60. How do you think about fairness and responsible AI?**
Check whether errors differ across groups, avoid features that proxy for protected attributes where they could harm people, document limits in a model card, keep a human in the loop for consequential decisions, and protect personal data. → [10.7](../../10-deployment-mlops/07-responsible-ai/), [09.11](../../09-nlp-genai/11-evaluating-llms/)

---

## Before the interview

- Pick two projects you can talk about for ten minutes each: the problem, your choices, the result vs baseline, what failed.
- Re-read your own READMEs. Interviewers ask about what you wrote.
- Prepare one question for them about how they evaluate and monitor models. It shows you know where projects go wrong.

**Next:** [11.5 What to learn next](../05-next-steps/).
