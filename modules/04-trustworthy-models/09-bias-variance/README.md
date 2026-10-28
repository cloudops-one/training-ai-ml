# 04.9 · Overfitting, underfitting and the bias–variance trade-off

⏱ about 1.5 hours · Prerequisites: 02.6 (first look at overfitting), 04.7 (cross-validation) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You have tried a lot of churn models now. Logistic regression sits stubbornly at about 68% accuracy. An unlimited decision tree scores **100% on the training customers** and about 60% on new ones. A KNN with *k* = 1 does the same. Your manager asks the reasonable question: *"So what do we do next? Get more data? Add features? Use a bigger model?"*

Those three actions fix **different** problems, and doing the wrong one wastes weeks. Collecting a year more data will not help a model that is too simple. Adding twenty features will make an overfitting model worse. You need a way to diagnose *which* problem you have before you choose the cure.

## 2. The idea

Picture four archers, each shooting many arrows at a target:

```
   low bias, low variance     high bias, low variance     low bias, high variance    high bias, high variance
        ┌─────────┐                ┌─────────┐                 ┌─────────┐                ┌─────────┐
        │         │                │  •••    │                 │ •     • │                │    •    │
        │   •••   │                │  •••    │                 │    ◎    │                │ ◎     • │
        │   •◎•   │                │    ◎    │                 │  •   •  │                │       • │
        │         │                │         │                 │ •    •  │                │  •      │
        └─────────┘                └─────────┘                 └─────────┘                └─────────┘
          ideal                    consistently off            right on average,           both
                                    (underfit)                 all over the place
                                                                 (overfit)
```

Each "arrow" is the same kind of model, trained on a different sample of customers.

- **Bias** is how far the *average* arrow lands from the bullseye. A model that is too simple misses in the same direction every time.
- **Variance** is how spread out the arrows are. A model that is too flexible lands somewhere different every time you retrain it, because it follows the quirks of each sample.

You cannot see your arrows directly; you only ever train on one sample. But you can see the symptoms: **training score vs cross-validated score**.

## 3. Definition

- **Underfitting (high bias)**: the model is too simple to capture the real pattern. It does poorly on training data *and* new data.
- **Overfitting (high variance)**: the model captures the pattern *plus* the noise of its particular training sample. Excellent on training data, much worse on new data.
- **Noise**: the part of the target no feature can explain (a customer's mood, a birthday, a broken laptop). No model can remove it.
- **Validation curve**: training and CV scores plotted against one complexity setting (tree depth, *k*, polynomial degree).
- **Learning curve**: training and CV scores plotted against the number of training rows.

## 4. Formula

For a new customer with true value $y$, the expected squared error of a model's prediction $\hat{y}$ (averaged over all the training samples it might have been trained on) splits exactly into three parts:

$$\underbrace{E\big[(y - \hat{y})^2\big]}_{\text{expected error}} = \underbrace{\big(E[\hat{y}] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{E\big[(\hat{y} - E[\hat{y}])^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$$

- $f(x)$: the true relationship. $E[\hat{y}]$: the model's average prediction over many retrains.
- **Bias²**: how far the average prediction is from the truth, squared.
- **Variance**: how much predictions scatter around their own average.
- $\sigma^2$: noise variance, the floor no model can go below.

Making a model more flexible lowers bias and raises variance. The best model minimises the **sum**.

## 5. By hand

A customer's true next-month spend is **£100**. Two models are each retrained on four different samples and asked to predict this customer (no noise, to keep it simple).

| retrain | Model A (simple) | Model B (flexible) |
|---|---|---|
| 1 | 90 | 120 |
| 2 | 92 | 80 |
| 3 | 88 | 110 |
| 4 | 90 | 90 |

**Model A.**

- Average prediction: (90 + 92 + 88 + 90)/4 = **90**.
- Bias: 90 − 100 = **−10**. Bias² = **100**.
- Deviations from 90: 0, 2, −2, 0. Squares: 0, 4, 4, 0. Variance = 8/4 = **2**.
- Errors from the truth: −10, −8, −12, −10. Squares: 100, 64, 144, 100. MSE = 408/4 = **102**.
- Check: bias² + variance = 100 + 2 = 102 ✓.

**Model B.**

- Average prediction: (120 + 80 + 110 + 90)/4 = **100**. Bias **0**.
- Deviations: 20, −20, 10, −10. Squares: 400, 400, 100, 100. Variance = 1000/4 = **250**.
- MSE = (400 + 400 + 100 + 100)/4 = **250** = 0 + 250 ✓.

Model B is "right on average" and still has more than twice the error. Being unbiased is not enough; being stable matters as much.

## 6. In code

Validation and learning curves are two scikit-learn functions:

```python
from sklearn.model_selection import validation_curve, learning_curve

train_scores, cv_scores = validation_curve(
    DecisionTreeClassifier(), X_train, y_train,
    param_name="max_depth", param_range=[1, 2, 3, 5, 8, 15], cv=5)

sizes, train_scores, cv_scores = learning_curve(
    model, X_train, y_train, train_sizes=[.1, .3, .5, .75, 1.0], cv=5)
```

Each returns one row per setting and one column per fold; take `.mean(axis=1)` and plot training against CV.

## 7. On real data

**Seeing bias and variance.** The notebook retrains polynomials of degree 1, 3 and 9 on 100 different samples of 30 customers drawn from a known curve. Averaged across the range:

| degree | bias² | variance | sum |
|---|---|---|---|
| 1 (line) | 179.6 | 52.2 | 231.8 |
| 3 | 0.5 | 50.5 | **51.0** |
| 9 | 7.2 | 2,924.8 | 2,931.9 |

**Tree depth on churn** (5-fold CV on 2,487 customers):

| max_depth | 1 | 3 | **5** | 8 | 15 | 20 |
|---|---|---|---|---|---|---|
| training accuracy | 0.647 | 0.681 | 0.711 | 0.770 | 0.937 | 0.983 |
| CV accuracy | 0.642 | 0.663 | **0.674** | 0.642 | 0.609 | 0.599 |

Up to depth 5 both scores rise together (bias falling). Beyond it the training score keeps climbing while CV falls (variance taking over).

**KNN on churn**: *k* = 1 scores 1.000 on training and 0.597 in CV, a 40-point gap. At *k* = 101 the gap is 0.005 and CV peaks at 0.682.

**Learning curves**: logistic regression's training and CV curves meet at **0.685**, so more of the same data will not help it. The unlimited tree stays at 1.000 training vs **0.602** CV at every size: a variance problem.

## 8. Where it breaks

| What you see | Problem | What helps | What does not |
|---|---|---|---|
| Train and CV both low, close together | **High bias** | More flexible model, better features, less regularisation | More data |
| Train high, CV much lower | **High variance** | Simpler model, regularisation (04.10), more data, fewer features, ensembles (Module 05) | More features |
| Train and CV both high, close | Good fit | Ship it, then monitor it (Module 10) | |

Cautions:

- **The decomposition is exact for squared error**, and a useful way of thinking for other metrics, but not a literal formula for accuracy or AUC.
- **Noise sets a ceiling.** If churn depends on things you do not record, no model reaches 100%. The logistic plateau at 0.685 may be close to what these five features allow.
- **Complexity is not only one knob.** Number of features, tree depth, polynomial degree, regularisation strength and training time all move the same balance.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: truth 50. Model C predicts 60, 62, 58, 60; model D predicts 40, 65, 45, 50. Compute bias², variance and MSE. Which is better?
2. Triple the noise in the simulation. Which polynomial degree wins now, and why?
3. Draw a validation curve for `min_samples_leaf` on the real churn data. Which end overfits?
4. Draw the learning curve for a depth-5 tree and compare its gap with the unlimited tree's.

**Quiz**

1. In one sentence each: what is bias, what is variance?
2. For model A in the by-hand table, what are bias², variance and MSE?
3. What is the "noise" term, and can a better model remove it?
4. Training accuracy 0.98, CV accuracy 0.61. Diagnosis and two remedies?
5. The training and CV learning curves have met at 0.685. Will doubling the data help?

## 10. Recap

- **Bias**: consistently wrong because too simple. **Variance**: unstable because too flexible. **Noise**: unavoidable.
- Expected squared error = **bias² + variance + noise**. Flexibility trades one for the other; minimise the sum.
- Diagnose with **training vs CV scores**: both low → bias; big gap → variance.
- **Validation curves** pick the complexity (tree depth 5, *k* ≈ 101 for ShopWave churn); **learning curves** tell you whether more data would help.
- More data cures variance, not bias. Better features cure bias, not variance.

**Next:** [04.10 Regularisation](../10-regularisation/) gives you a smooth knob for turning variance down without changing the model type.
