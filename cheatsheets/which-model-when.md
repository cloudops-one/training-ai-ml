# Which model when

Start with the **baseline**, then the **first real model**, and only reach for the heavy options if they beat both on a fair test (cross-validation or a time-ordered backtest). Numbers in *italics* are what the course actually measured on ShopWave data.

## Decision table

| Problem | Baseline to beat | First real model | Then try | Wins when… | Loses when… | Key knobs | Learn it |
|---|---|---|---|---|---|---|---|
| **Regression** (predict a number: spend next quarter) | Predict the mean or median | Linear regression on scaled features | Ridge/Lasso, gradient boosting (LightGBM/XGBoost), decision tree | Linear: relationship roughly additive, few rows. Boosting: many interactions, lots of rows. | Linear: strong curves or thresholds. Trees: small data (overfit), extrapolation. | `alpha` (Ridge/Lasso), `max_depth`, `learning_rate`, `n_estimators` | [02.2](../modules/02-regression/02-linear-regression-by-hand/), [02.5](../modules/02-regression/05-multiple-regression/), [04.10](../modules/04-trustworthy-models/10-regularisation/) |
| **Binary classification** (churn yes/no) | Majority class; one-rule ("no order for 120 days") | Logistic regression in a Pipeline | Random forest, gradient boosting, KNN | Logistic: signal is smooth and additive. Trees: thresholds and interactions. | Logistic: cliffs/thresholds. Boosting: small or noisy data, untuned defaults. | `C` (logistic), `n_estimators`, `max_depth`/`num_leaves`, `min_samples_leaf` | [03.4](../modules/03-classification/04-logistic-regression/), [05.2](../modules/05-powerful-models/02-random-forest/), [05.4](../modules/05-powerful-models/04-xgboost-lightgbm/) |
| **Multi-class** (segment, product type) | Most common class | Multinomial logistic regression | Random forest, boosting, (images/text: neural nets) | Same as binary | Rare classes get ignored → `class_weight="balanced"` | as above | [03.8](../modules/03-classification/08-multiclass/) |
| **Imbalanced** (fraud, top-5% spenders) | "Predict nobody" (looks 95% accurate) | Logistic or boosting **with `class_weight`** + tuned threshold | Resampling / SMOTE **inside** an imblearn Pipeline | Threshold chosen by business cost beats any resampling trick | SMOTE before the split leaks (*PR-AUC 0.992 fake vs 0.579 honest*) | `class_weight`, threshold, `scale_pos_weight` | [05.7](../modules/05-powerful-models/07-imbalanced-data/), [03.7](../modules/03-classification/07-roc-auc-threshold/) |
| **Clustering** (customer segments) | Business rules (RFM bands) | K-Means on log-transformed, scaled features | Hierarchical (Ward), DBSCAN, Gaussian mixture | K-Means: round-ish groups of similar size. DBSCAN: odd shapes + noise. GMM: soft membership. | K-Means on unscaled money columns (one cluster takes everyone) | `k`, `eps`/`min_samples`, linkage | [06.2](../modules/06-unsupervised/02-kmeans-by-hand/), [06.3](../modules/06-unsupervised/03-choosing-k/), [06.5](../modules/06-unsupervised/05-dbscan-gmm/) |
| **Anomaly detection** (odd invoices) | z-score / IQR rule per column | Isolation Forest | Local Outlier Factor, robust (median/MAD) z-score | Many columns, no labels | Expecting "anomaly = fraud": *44% of flags were postage/manual lines* | `contamination`, `n_estimators` | [06.8](../modules/06-unsupervised/08-anomaly-detection/) |
| **What sells together** | Most popular items | Apriori + association rules | FP-Growth | Read rules by **lift**, not confidence | Popular items (POSTAGE) create high-confidence, meaningless rules | `min_support`, `min_lift` | [06.9](../modules/06-unsupervised/09-market-basket/) |
| **Recommendation** | Top sellers for everyone | Item-item collaborative filtering (cosine) | Content-based (embeddings), matrix factorisation | Customer has history | Cold start (new customer or product) → fall back to popularity | number of neighbours, top-N products | [06.10](../modules/06-unsupervised/10-recommenders/) |
| **Forecasting** (weekly revenue) | **Seasonal naive** (same week last year) | Holt-Winters, SARIMA | Gradient boosting on lag/calendar features | Long history with stable seasons | Only ~2 seasons of history: fancy models collapse to seasonal naive | `seasonal_periods`, `(p,d,q)(P,D,Q,s)`, lags | [07.2](../modules/07-forecasting/02-baselines-and-evaluation/), [07.3](../modules/07-forecasting/03-smoothing-holt-winters/), [07.5](../modules/07-forecasting/05-ml-forecasting/) |
| **Images** (tag product photos) | Logistic on raw pixels | Small CNN | **Transfer learning** (frozen MobileNet/EfficientNet + small head), then fine-tune | Few labelled photos → transfer learning by a mile | Training a CNN from scratch on 1,000 photos | learning rate, epochs + EarlyStopping, augmentation | [08.8](../modules/08-deep-learning/08-cnns-fashion-mnist/), [08.9](../modules/08-deep-learning/09-augmentation-transfer-learning/) |
| **Text classification** (review sentiment, email routing) | Majority class; keyword rules | TF-IDF + logistic regression | Sentence embeddings + logistic; fine-tuned DistilBERT; zero-shot / LLM prompt | TF-IDF: thousands of labels. Embeddings: few labels. Fine-tuning: best accuracy. LLM: no labels at all. | TF-IDF: negation, sarcasm, synonyms | `ngram_range`, `C`, `max_length`, epochs | [09.2](../modules/09-nlp-genai/02-sentiment-classifier/), [09.3](../modules/09-nlp-genai/03-embeddings/), [09.6](../modules/09-nlp-genai/06-fine-tuning/) |
| **Generation / Q&A over your documents** | Plain LLM prompt (hallucinates your policies) | **RAG**: chunk → embed → retrieve top-k → answer with citations | Better chunking, re-ranking, an agent with tools | Facts live in your documents and change | Questions the documents do not cover → must say "I don't know" | chunk size/overlap, `k`, prompt | [09.10](../modules/09-nlp-genai/10-rag/), [09.11](../modules/09-nlp-genai/11-evaluating-llms/), [09.12](../modules/09-nlp-genai/12-agents-tool-calling/) |

## What the course actually found

| Task | Result | Lesson |
|---|---|---|
| ShopWave churn (tabular) | Logistic regression *AUC 0.741*; tuned LightGBM *0.742*; neural net *0.740*. A tie, so the simpler model was chosen. | [05.10](../modules/05-powerful-models/10-choosing-and-saving/) |
| Synthetic fraud with threshold rules | Logistic *0.815* vs random forest *0.885*, LightGBM *0.874*. Trees win on cliffs. | [05.4](../modules/05-powerful-models/04-xgboost-lightgbm/) |
| Spend next quarter | Six-feature linear model beat every tree (*test R² 0.651*). | [02.P](../modules/02-regression/project/) |
| Weekly revenue | Seasonal naive *£32.5k* MAE; Holt-Winters *£34.7k*; SARIMA *£34.6k*; recursive boosting *£30.2k* (small, suggestive win). | [07.5](../modules/07-forecasting/05-ml-forecasting/) |
| Recommendations | Collaborative filtering hit rate@10 *30.0%* vs popularity *20.3%*. | [06.10](../modules/06-unsupervised/10-recommenders/) |
| Fashion-MNIST | Dense net *0.873* vs small CNN *0.896* with fewer parameters. | [08.8](../modules/08-deep-learning/08-cnns-fashion-mnist/) |
| 1,000 flower photos | From scratch *0.552* vs frozen MobileNetV3 *0.846*. | [08.9](../modules/08-deep-learning/09-augmentation-transfer-learning/) |
| Review sentiment, 2,000 labels | Zero-shot *81.7%*, TF-IDF *84.7%*, fine-tuned DistilBERT *90.6%*. | [09.6](../modules/09-nlp-genai/06-fine-tuning/) |

## Rules of thumb

1. **Always print the baseline first.** If your model does not beat it on a fair test, you have no model.
2. **Tabular data → logistic/linear and gradient boosting.** Neural nets rarely win here.
3. **Images, text, audio → pretrained deep models.** Do not start from scratch with small data.
4. **When two models tie (within one CV standard deviation), pick the simpler, faster, explainable one.** → [05.10](../modules/05-powerful-models/10-choosing-and-saving/)
5. **Split by time when the future is what you predict.** → [04.1](../modules/04-trustworthy-models/01-train-validation-test/)
6. **Tuning repairs bad defaults; it rarely transforms a model.** → [05.6](../modules/05-powerful-models/06-hyperparameter-tuning/)
7. **Choose the threshold from business costs, not 0.5.** → [03.7](../modules/03-classification/07-roc-auc-threshold/)
8. **AutoML (PyCaret) is a fast leaderboard, not a substitute for understanding it.** → [05.9](../modules/05-powerful-models/09-automl-pycaret/)
