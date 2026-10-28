# Metrics

$y$ = true value, $\hat{y}$ = prediction, $n$ = number of rows, $\bar{y}$ = mean of $y$. ↑ = higher is better, ↓ = lower is better.

## Regression

| Metric | Formula | Range | Better | Use it when | Watch out | Learn it |
|---|---|---|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y-\hat{y}\rvert$ | 0 → ∞, target units | ↓ | Explaining error to the business ("off by £547 on average") | Treats a £10 miss on a small and a big customer the same | [02.3](../modules/02-regression/03-measuring-error/) |
| **MSE** | $\frac{1}{n}\sum (y-\hat{y})^2$ | 0 → ∞, units² | ↓ | Training loss (smooth, easy to differentiate) | Squared units; one outlier dominates | [02.3](../modules/02-regression/03-measuring-error/) |
| **RMSE** | $\sqrt{\text{MSE}}$ | 0 → ∞, target units | ↓ | Big misses are much worse than small ones | Always ≥ MAE; the gap shows how spiky errors are | [02.3](../modules/02-regression/03-measuring-error/) |
| **R²** | $1 - \frac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2}$ | −∞ → 1 | ↑ | "Share of variation explained"; comparing to predicting the mean (R² = 0) | Negative on test data = worse than the mean; one wholesaler took R² from 0.98 to −0.29 | [02.3](../modules/02-regression/03-measuring-error/) |
| **MAPE** | $\frac{100}{n}\sum \left\lvert\frac{y-\hat{y}}{y}\right\rvert$ | 0 → ∞ % | ↓ | Comparing series of different sizes, if no value is near zero | Explodes near zero (*969% while MAE was £16k*) | [07.2](../modules/07-forecasting/02-baselines-and-evaluation/) |
| **sMAPE** | $\frac{100}{n}\sum \frac{\lvert y-\hat{y}\rvert}{(\lvert y\rvert+\lvert\hat{y}\rvert)/2}$ | 0 → 200 % | ↓ | A less explosive MAPE | Asymmetric: over- and under-forecasts are punished differently | [07.2](../modules/07-forecasting/02-baselines-and-evaluation/) |
| **MASE** | $\frac{\text{MAE}}{\frac{1}{T-m}\sum_{t=m+1}^{T}\lvert y_t-y_{t-m}\rvert}$ | 0 → ∞ | ↓ (< 1 beats seasonal naive) | Forecasts: "better than copying last season?" | Needs a training history at least one season long ($m$) | [07.2](../modules/07-forecasting/02-baselines-and-evaluation/) |

## Classification

Confusion matrix: **TP** caught churners, **FP** false alarms, **FN** missed churners, **TN** correctly left alone.

| Metric | Formula | Range | Better | Use it when | Watch out | Learn it |
|---|---|---|---|---|---|---|
| **Accuracy** | $\frac{TP+TN}{TP+FP+FN+TN}$ | 0 → 1 | ↑ | Classes roughly balanced | "Predict nobody" scores 95% at 5% positives | [03.6](../modules/03-classification/06-confusion-matrix-metrics/) |
| **Precision** | $\frac{TP}{TP+FP}$ | 0 → 1 | ↑ | False alarms are expensive (each one costs a voucher) | Can be high while missing most positives | [03.6](../modules/03-classification/06-confusion-matrix-metrics/) |
| **Recall** | $\frac{TP}{TP+FN}$ | 0 → 1 | ↑ | Missing a positive is expensive (fraud, churn) | Flag everyone → recall 1.0 | [03.6](../modules/03-classification/06-confusion-matrix-metrics/) |
| **F1** | $2\cdot\frac{P\cdot R}{P+R}$ | 0 → 1 | ↑ | One number balancing precision and recall | Ignores TN; costs are rarely equal → prefer an expected-cost calculation | [03.6](../modules/03-classification/06-confusion-matrix-metrics/) |
| **Macro / weighted F1** | mean of per-class F1 (plain / weighted by class size) | 0 → 1 | ↑ | Multi-class; macro treats rare classes as equally important | Weighted hides a failing small class | [03.8](../modules/03-classification/08-multiclass/) |
| **ROC-AUC** | area under TPR vs FPR over all thresholds; = P(random positive scored above random negative) | 0.5 (random) → 1 | ↑ | Ranking quality, threshold-free model comparison | Looks good even on heavily imbalanced data | [03.7](../modules/03-classification/07-roc-auc-threshold/) |
| **PR-AUC** (average precision) | area under precision vs recall | base rate → 1 | ↑ | Rare positives (fraud, top-5% spenders) | Baseline is the positive rate, not 0.5 | [05.7](../modules/05-powerful-models/07-imbalanced-data/) |
| **Log loss** (binary cross-entropy) | $-\frac{1}{n}\sum \left[y\ln p + (1-y)\ln(1-p)\right]$ | 0 → ∞ | ↓ | Quality of the probabilities themselves; training loss | Confident and wrong is punished hard (*2.303 for p = 0.1 when y = 1*) | [03.4](../modules/03-classification/04-logistic-regression/), [08.3](../modules/08-deep-learning/03-activations-and-losses/) |
| **Expected cost** | $FP\cdot c_{FP} + FN\cdot c_{FN}$ | £ | ↓ | Choosing the threshold (*£5 voucher vs £60 lost customer*) | Needs honest cost estimates from the business | [03.7](../modules/03-classification/07-roc-auc-threshold/) |

## Unsupervised and ranking

| Metric | Formula | Range | Better | Use it when | Watch out | Learn it |
|---|---|---|---|---|---|---|
| **Inertia** | $\sum_i \lVert x_i - \mu_{c(i)}\rVert^2$ | 0 → ∞ | ↓ | Elbow plot for K-Means | Always falls as $k$ grows | [06.2](../modules/06-unsupervised/02-kmeans-by-hand/) |
| **Silhouette** | $s = \frac{b-a}{\max(a,b)}$, $a$ = mean distance to own cluster, $b$ = to nearest other | −1 → 1 | ↑ | Picking $k$; spotting badly placed points | Favours round, separated blobs; business sense can overrule it (*k = 2 best, k = 4 chosen*) | [06.3](../modules/06-unsupervised/03-choosing-k/) |
| **Davies-Bouldin** | mean over clusters of $\max_j \frac{s_i+s_j}{d(\mu_i,\mu_j)}$ | 0 → ∞ | ↓ | Second opinion on $k$ | Same blob bias as silhouette | [06.3](../modules/06-unsupervised/03-choosing-k/) |
| **Support / confidence / lift** | $P(A\cap B)$; $P(B\mid A)$; $\frac{P(B\mid A)}{P(B)}$ | 0 → 1; 0 → 1; 0 → ∞ | lift > 1 | Market basket rules | High confidence with lift ≈ 1 means "B is just popular" | [06.9](../modules/06-unsupervised/09-market-basket/) |
| **Hit rate @k** | share of customers whose held-out next purchase is in their top-$k$ list | 0 → 1 | ↑ | Recommenders | Compare against popularity at the same $k$ | [06.10](../modules/06-unsupervised/10-recommenders/) |
| **Retrieval hit @k** | share of questions whose correct document is in the top-$k$ chunks | 0 → 1 | ↑ | RAG retrieval quality | Separate from answer quality; check both | [09.11](../modules/09-nlp-genai/11-evaluating-llms/) |
| **PSI** (population stability index) | $\sum_b (p_b^{new}-p_b^{old})\ln\frac{p_b^{new}}{p_b^{old}}$ over bins $b$ | 0 → ∞ | ↓ (< 0.1 stable, 0.1–0.25 watch, > 0.25 shifted) | Data drift monitoring in production | Bins with zero counts need a small floor | [10.6](../modules/10-deployment-mlops/06-monitoring-drift/) |

## Which metric to report

| Audience | Report |
|---|---|
| Business | MAE in £, precision/recall at the chosen threshold, expected cost, "of the 500 flagged, N churned" |
| Model comparison | Cross-validated AUC (classification), MAE/RMSE (regression), backtest MAE (forecasting), mean ± std |
| Rare positives | PR-AUC and recall at a fixed precision, never plain accuracy |

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, log_loss,
                             confusion_matrix, classification_report, silhouette_score)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])   # scores, not labels
```
